import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, pack
from torch import Tensor, BoolTensor

from .dit_encoder import DiTEncoder
from .layers import FinalLinear, ScalarEmbedder
from ..conv.conv_layers import PostNet, ResBlock1d


class DiTSpeechEncoder(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        spk_dim: int,
        output_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        postnet_mult: int,
        postnet_dim: int,
        dropout_rate: float = 0.1,
        cond_cfg_rate: float = 0.25,
        spk_cfg_rate: float = 0.25,
    ):
        super().__init__()
        self.prenet = ResBlock1d(cond_dim + output_dim + spk_dim, embed_dim, embed_dim)
        self.t_encoder = ScalarEmbedder(256, embed_dim // 2)
        self.spk_proj = nn.Linear(spk_dim, embed_dim - embed_dim // 2)
        self.dit = DiTEncoder(embed_dim, embed_dim * 4, num_heads, num_layers, dropout_rate)
        self.final_linear = FinalLinear(embed_dim, output_dim)
        if postnet_mult > 0:
            self.postnet = PostNet(postnet_mult, postnet_dim, output_dim)
        else:
            self.postnet = None

        self.output_dim = output_dim
        self.cond_cfg_rate = cond_cfg_rate
        self.spk_cfg_rate = spk_cfg_rate

    def forward(
        self,
        x: Tensor,
        p: Tensor,
        t: Tensor,
        mask: BoolTensor,
        cond: Tensor,
        spk_emb: Tensor,
    ) -> Tensor:
        """
        Args:
            x (Tensor): [N, ..., T_dec, D_out], input feature sequence.
            p (Tensor): [N, ..., T_dec], the decoder position tensor.
            t (Tensor): [N, ...], the time tensor.
            mask (BoolTensor): [N, T_dec], feature mask.
            cond (Tensor): [N, ..., T_cond, D_cond], encoder output condition sequence.
            spk_emb (Tensor): [N, ..., D_spk], speaker embedding.
        Returns:
            y (Tensor): [N, ..., T_dec, D_out].
        """
        x_res = x

        # NOTE: Timbre-aware time modulation
        t_emb = self.t_encoder.forward(t)  # [N, D/2]
        spk_emb_adaln = self.spk_proj.forward(spk_emb)  # [N, D/2]
        t_emb, _ = pack([t_emb, spk_emb_adaln], "b *")  # [N, D]

        spk_cond = repeat(spk_emb, "n d -> n t d", t=cond.shape[1])  # [N, T_cond, D_spk]
        x_with_cond, _ = pack([cond, spk_cond, x], "b t *")  # [N, T_cond, D_cond + D_spk + D_out]
        x_with_cond = self.prenet(x_with_cond, mask, t_emb)

        N, T_feat = mask.shape
        attn_mask = mask.unsqueeze(1).expand(-1, T_feat, -1)  # [N, T_feat, T_feat]
        x = self.dit(x_with_cond, p, t_emb, attn_mask)  # [N, T_feat, D]

        x = self.final_linear(x, t_emb) * mask.unsqueeze(-1)
        if self.postnet is not None:
            x = self.postnet(x, x_res, mask)

        return x

    def make_positions(self, length: int, feat: Tensor) -> Tensor:
        _p = torch.arange(0, length, 1, dtype=feat.dtype, device=feat.device)  # [T]
        p = repeat(_p, "t -> b t", b=feat.shape[0])  # [B, T]
        return p

    @torch.inference_mode()
    def inference(
        self,
        mask: BoolTensor,
        cond: Tensor,
        spk_emb: Tensor,
        n_timesteps: int = 16,
        temperature: float = 1.0,
        t_scheduler: str = "cosine",
        full_cfg: float = 1.0,
        cond_cfg: float = 0.0,
        spk_cfg: float = 0.0,
        sde_noise_scale: float = 0.0,
        decay_noise: bool = True,
    ):
        """
        Args:
            mask (BoolTensor): [N, T_dec], input feature mask.
            cond (Tensor): [N, T_cond, D_cond], encoder output condition sequence.
            spk_emb (Tensor): [N, D_spk], speaker embedding.
            n_timesteps (int): number of time steps for inference.
            temperature (float): temperature for sampling.
            t_scheduler (str): time scheduler. Defaults to "cosine".
            full_cfg (float): initial strength of the full CFG.
            cond_cfg (float): initial strength of the CFG.
            cond_cfg_decay (str): decay type for CFG strength. Defaults to "cosine".
            cond_cfg_min (float): minimum CFG scale factor at t=1.
            sde_noise_scale (float): scaling factor for the noise term.
            decay_noise (bool): whether noise magnitude should decay toward t=1.
        Returns:
            x (FloatTensor): [N, T_dec, D_out], output feature sequence.
        """
        N, T = mask.shape
        D = self.output_dim
        x = torch.randn(N, T, D, device=cond.device, dtype=cond.dtype) * temperature
        p = self.make_positions(T, x)

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=cond.device, dtype=cond.dtype)
        if t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            # Get the velocity field prediction
            t_ = t.unsqueeze(dim=0).expand(N)  # [N]
            dphi_dt = self.forward(x, p, t_, mask, cond, spk_emb)

            # Apply classifier-free guidance if needed
            if cond_cfg != 0:
                cond_cfg_dphi_dt = self.forward(x, p, t_, mask, torch.zeros_like(cond), spk_emb)
            else:
                cond_cfg_dphi_dt = torch.zeros_like(dphi_dt)

            if spk_cfg != 0:
                spk_cfg_dphi_dt = self.forward(x, p, t_, mask, cond, torch.zeros_like(spk_emb))
            else:
                spk_cfg_dphi_dt = torch.zeros_like(dphi_dt)

            if full_cfg != 0:
                full_cfg_dphi_dt = self.forward(x, p, t_, mask, torch.zeros_like(cond), torch.zeros_like(spk_emb))
            else:
                full_cfg_dphi_dt = torch.zeros_like(dphi_dt)

            # 2-way classifier-free guidance
            dphi_dt = (
                (1 + cond_cfg + full_cfg + spk_cfg) * dphi_dt
                - cond_cfg * cond_cfg_dphi_dt
                - full_cfg * full_cfg_dphi_dt
                - spk_cfg * spk_cfg_dphi_dt
            )

            # Deterministic update
            x_update = dt * dphi_dt

            # Stochastic update (if SDE mode enabled)
            if sde_noise_scale > 0:
                # Calculate noise coefficient that can decay over time if requested
                if decay_noise:
                    # Higher noise at beginning, lower at end
                    time_factor = 1.0 - t.item()
                else:
                    time_factor = 1.0

                # Scale noise by time factor, overall scale, and sqrt of step size
                noise_coeff = sde_noise_scale * time_factor * torch.sqrt(torch.as_tensor(dt, device=x.device))

                # Generate and add the noise term
                noise = torch.randn_like(x) * noise_coeff
                x_update = x_update + noise

            # Update x
            x = x + x_update

            # Update time
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return x

    def compute_loss(
        self,
        x1: Tensor,
        mask: BoolTensor,
        cond: Tensor,
        spk_emb: Tensor,
        t_scheduler: str = "cosine",
        sigma_min: float = 1e-6,
    ) -> Tensor:
        """
        Args: Same as forward(...)
            x1 (FloatTensor): [N, T_dec, D_out], input feature sequence.
            feat_mask (BoolTensor): [N, T_dec], input feature mask.
            cond (FloatTensor): [N, T_cond, D_cond], encoder output condition sequence.
            spk_emb (FloatTensor): [N, D_spk], speaker embedding.
            t_scheduler (str): time scheduler. Defaults to "cosine".
            sigma_min (float): minimum noise level. Defaults to 1e-6.
        Returns:
            loss (Tensor): [1], loss value.
        """
        B, T, D = x1.shape
        t = torch.rand([B], device=x1.device, dtype=x1.dtype)
        if t_scheduler == "cosine":
            t = 1 - torch.cos(t * 0.5 * math.pi)
        _t = t.view(-1, 1, 1)
        z = torch.randn_like(x1)

        y = (1 - (1 - sigma_min) * _t) * z + _t * x1
        u = x1 - (1 - sigma_min) * z

        p = self.make_positions(T, x1)

        # Apply classifier-free guidance
        # cfg_prob = torch.rand(B, device=cond.device)
        # True -> conditional training; False -> unconditional training
        if self.cond_cfg_rate > 0:
            cond_cfg_prob = torch.rand(B, device=cond.device)
            cond_cfg_mask = cond_cfg_prob > self.cond_cfg_rate
            cond = cond * cond_cfg_mask.view(-1, 1, 1)

        if self.spk_cfg_rate > 0:
            spk_cfg_prob = torch.rand(B, device=cond.device)
            spk_cfg_mask = spk_cfg_prob > self.spk_cfg_rate
            spk_emb = spk_emb * spk_cfg_mask.view(-1, 1)

        pred = self.forward(y, p, t, mask, cond, spk_emb)
        loss = F.mse_loss(pred * mask.unsqueeze(-1), u * mask.unsqueeze(-1), reduction="sum") / (torch.sum(mask) * D)

        return loss
