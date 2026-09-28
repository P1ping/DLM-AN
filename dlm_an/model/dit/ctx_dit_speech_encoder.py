import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat, pack
from torch import Tensor, BoolTensor

from .dit_encoder import DiTEncoder
from .layers import FinalLinear, ScalarEmbedder
from ..conv.conv_layers import PostNet, ResBlock1d


class CtxDiTSpeechEncoder(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        output_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        postnet_mult: int,
        postnet_dim: int,
        spk_dim: int = 256,
        dropout_rate: float = 0.1,
        cond_cfg_rate: float = 0.25,
        spk_cfg_rate: float = 0.25,
    ):
        super().__init__()
        self.prenet = ResBlock1d(cond_dim + output_dim * 2 + spk_dim, embed_dim, embed_dim)
        self.use_spk_embed = spk_dim > 0
        if spk_dim > 0:
            # NOTE: Timbre-aware time modulation
            self.t_encoder = ScalarEmbedder(256, embed_dim // 2)
            self.spk_proj = nn.Linear(spk_dim, embed_dim - embed_dim // 2)
        else:
            self.t_encoder = ScalarEmbedder(256, embed_dim)
            self.spk_proj = None
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
        x_ctx: Tensor,
        spk_emb: Tensor = None,
    ) -> Tensor:
        """
        Args:
            x (Tensor): [N, ..., T_dec, D_out], input feature sequence.
            p (Tensor): [N, ..., T_dec], the decoder position tensor.
            t (Tensor): [N, ...], the time tensor.
            mask (BoolTensor): [N, T_dec], feature mask.
            cond (Tensor): [N, ..., T_cond, D_cond], encoder output condition sequence.
            x_ctx (Tensor): [N, ..., D_spk], context feature sequence.
            spk_emb (Tensor): [N, D_spk], speaker embedding.
        Returns:
            y (Tensor): [N, ..., T_dec, D_out].
        """
        x_res = x

        t_emb = self.t_encoder.forward(t)  # [N, D] or [N, D/2]
        if self.use_spk_embed:
            spk_emb_adaln = self.spk_proj(spk_emb)  # [N, D/2]
            t_emb, _ = pack([t_emb, spk_emb_adaln], "b *")  # [N, D]

        if self.use_spk_embed:
            spk_cond = repeat(spk_emb, "n d -> n t d", t=cond.shape[1])  # [N, T_cond, D_spk]
            x_with_cond, _ = pack([cond, spk_cond, x_ctx, x], "b t *")  # [N, T_cond, D_cond + D_spk + D_out + D_out]
        else:
            x_with_cond, _ = pack([cond, x_ctx, x], "b t *")  # [N, T_cond, D_cond + D_out + D_out]
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
        ctx: Tensor,
        ctx_cond: Tensor,
        ctx_mask: BoolTensor,
        spk_emb: Tensor = None,
        n_timesteps: int = 16,
        temperature: float = 1.0,
        t_scheduler: str = "cosine",
        full_cfg: float = 1.0,
        cond_cfg: float = 0.0,
        spk_cfg: float = 0.0,
    ):
        """
        Args:
            mask (BoolTensor): [N, T_dec], input feature mask.
            cond (Tensor): [N, T_cond, D_cond], encoder output condition sequence.
            ctx (Tensor): [N, T_dec, D_out], context feature sequence.
            ctx_cond (Tensor): [N, T_dec, D_cond], context condition sequence.
            ctx_mask (BoolTensor): [N, T_dec], context mask.
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
            x_target (FloatTensor): [N, T_dec, D_out], output feature sequence.
        """
        total_cond, total_ctx, total_mask, target_mask = self.concatenate_context(cond, mask, ctx, ctx_cond, ctx_mask)

        N, T, D_cond = total_cond.shape
        D = self.output_dim

        x = torch.randn(N, T, D, device=cond.device, dtype=cond.dtype) * temperature
        x = x * target_mask.unsqueeze(-1)  # Zero out the context part of `x` for training-inference consistency
        p = self.make_positions(T, x)

        spk_emb_cfg = torch.zeros_like(spk_emb) if self.use_spk_embed else None

        t_span = torch.linspace(0, 1, n_timesteps + 1, device=cond.device, dtype=cond.dtype)
        if t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        for step in range(1, len(t_span)):
            # Get the velocity field prediction
            t_ = t.unsqueeze(dim=0).expand(N)  # [N]
            dphi_dt = self.forward(x, p, t_, total_mask, total_cond, total_ctx, spk_emb)

            # Apply classifier-free guidance if needed
            if cond_cfg != 0:
                cond_cfg_dphi_dt = self.forward(x, p, t_, total_mask, torch.zeros_like(total_cond), total_ctx, spk_emb)
            else:
                cond_cfg_dphi_dt = torch.zeros_like(dphi_dt)

            if spk_cfg != 0:
                spk_cfg_dphi_dt = self.forward(
                    x, p, t_, total_mask, total_cond, torch.zeros_like(total_ctx), spk_emb_cfg
                )
            else:
                spk_cfg_dphi_dt = torch.zeros_like(dphi_dt)

            if full_cfg != 0:
                full_cfg_dphi_dt = self.forward(
                    x, p, t_, total_mask, torch.zeros_like(total_cond), torch.zeros_like(total_ctx), spk_emb_cfg
                )
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

            # Update x -- we only update the non-context part of `x`, while keeping the context part fixed to the input `x_ctx` (zero)
            x[target_mask] = x[target_mask] + x_update[target_mask]

            # Update time
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        x_target = self.collect_non_context(x, target_mask)

        return x_target

    def concatenate_context(
        self,
        cond,
        mask,
        ctx,
        ctx_cond,
        ctx_mask,
    ):
        B, T_cond_max, D_cond = cond.shape
        _, T_ctx_max, D_out = ctx.shape

        ctx_len = ctx_mask.sum(dim=1)  # [N]
        x_len = mask.sum(dim=1)  # [N]
        total_len = ctx_len + x_len  # [N]
        T = total_len.max().item()

        device = cond.device
        dtype = cond.dtype

        positions = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        total_mask = positions < total_len.unsqueeze(1)  # [N, T]

        non_ctx_mask = total_mask & (positions >= ctx_len.unsqueeze(1))  # [N, T]
        total_cond = torch.zeros(B, T, D_cond, device=device, dtype=dtype)
        total_ctx = torch.zeros(B, T, D_out, device=device, dtype=dtype)

        # Initialize safe-sized tensors with zeros
        T_safe = max(T, (ctx_len + T_cond_max).max().item())
        total_cond_safe = torch.zeros(B, T_safe, D_cond, device=device, dtype=dtype)
        total_ctx_safe = torch.zeros(B, T_safe, D_out, device=device, dtype=dtype)

        # Scatter the context features at the beginning (indices 0 to T_ctx_max - 1)
        idx_ctx_cond = torch.arange(T_ctx_max, device=device).view(1, -1, 1).expand(B, -1, D_cond)
        total_cond_safe.scatter_(1, idx_ctx_cond, ctx_cond)

        idx_ctx = torch.arange(T_ctx_max, device=device).view(1, -1, 1).expand(B, -1, D_out)
        total_ctx_safe.scatter_(1, idx_ctx, ctx)

        # Scatter the target condition features starting at ctx_len
        # We offset the indices by ctx_len for each batch element.
        # The valid data of `cond` will perfectly overwrite the padding zeros of `ctx_cond`.
        idx_cond = torch.arange(T_cond_max, device=device).unsqueeze(0) + ctx_len.unsqueeze(1)
        idx_cond = idx_cond.unsqueeze(-1).expand(B, -1, D_cond)
        total_cond_safe.scatter_(1, idx_cond, cond)

        # Truncate back to the exact max valid sequence length T
        total_cond = total_cond_safe[:, :T, :]
        total_ctx = total_ctx_safe[:, :T, :]

        return total_cond, total_ctx, total_mask, non_ctx_mask

    def collect_non_context(self, x, non_ctx_mask):
        """
        Extracts the generated target part from the concatenated sequence.

        Args:
            x: [B, T, D_out] - The full sequence (context + target + padding)
            non_ctx_mask: [B, T] - Boolean mask where True indicates the target part

        Returns:
            x_target: [B, T_target_max, D_out] - The extracted target sequences,
                    shifted to start at index 0, with padding applied.
        """
        B, T, D_out = x.shape

        x_lens = non_ctx_mask.sum(dim=1)  # [B]
        T_target_max = x_lens.max().item()

        # Edge case: if there is no target part to collect
        if T_target_max == 0:
            return torch.zeros(B, 0, D_out, device=x.device, dtype=x.dtype)

        # Find the starting index of the target part for each sequence
        # argmax on a boolean tensor returns the index of the first True value
        start_indices = non_ctx_mask.int().argmax(dim=1)  # [B]

        # Create indices for gathering
        # Base indices: [1, T_target_max]
        # Shift by start_indices: [B, 1]
        # Resulting idx: [B, T_target_max]
        idx = torch.arange(T_target_max, device=x.device).unsqueeze(0) + start_indices.unsqueeze(1)

        # Clamp indices to avoid out-of-bounds errors for sequences shorter than T_target_max
        # (The invalid gathered values will be zeroed out in step 5)
        idx = idx.clamp(max=T - 1)

        # Expand indices to match the feature dimension: [B, T_target_max, D_out]
        idx = idx.unsqueeze(-1).expand(-1, -1, D_out)

        # Gather the target parts
        x_target = torch.gather(x, 1, idx)

        # Mask out the padding elements that were gathered due to clamping
        # Create a mask for the valid target lengths: [B, T_target_max]
        target_mask = torch.arange(T_target_max, device=x.device).unsqueeze(0) < x_lens.unsqueeze(1)

        # Apply the mask
        x_target = x_target * target_mask.unsqueeze(-1)

        return x_target

    def compute_loss(
        self,
        x1: Tensor,
        mask: BoolTensor,
        cond: Tensor,
        spk_emb: Tensor = None,
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

        # Select a random-lengthed prefix of `x1` as `ctx` (speaker prompt)
        x_lens = mask.sum(dim=1)  # [N]
        beta_dist = torch.distributions.Beta(3.0, 7.0)  # Bias towards shorter prefixes
        ratios = beta_dist.sample([B]).to(device=x1.device, dtype=x1.dtype)
        ctx_lens = (ratios * x_lens.float()).long().clamp(min=0)  # [N]
        positions = torch.arange(T, device=x1.device).unsqueeze(0)  # [1, T]
        ctx_mask = positions < ctx_lens.unsqueeze(1)  # [N, T]
        ctx = x1 * ctx_mask.unsqueeze(-1)  # [N, T, D_out], zeros beyond prefix
        y_ = y * (~ctx_mask).unsqueeze(-1) * mask.unsqueeze(-1)  # Zero out `y` in the prompt region
        loss_mask = mask & ~ctx_mask  # Only compute loss on non-prompt part

        # Apply classifier-free guidance
        # True -> conditional training; False -> unconditional training
        if self.cond_cfg_rate > 0:
            cond_cfg_prob = torch.rand(B, device=cond.device)
            cond_cfg_mask = cond_cfg_prob > self.cond_cfg_rate
            cond = cond * cond_cfg_mask.view(-1, 1, 1)

        if self.spk_cfg_rate > 0:
            spk_cfg_prob = torch.rand(B, device=cond.device)
            spk_cfg_mask = spk_cfg_prob > self.spk_cfg_rate
            ctx = ctx * spk_cfg_mask.view(-1, 1, 1)
            spk_emb = spk_emb * spk_cfg_mask.view(-1, 1) if spk_emb is not None else None
            # loss_mask = (mask & spk_cfg_mask.view(-1, 1)) | ctx_mask  # Include prompt in loss if spk_cfg is applied

        pred = self.forward(y_, p, t, mask, cond, ctx, spk_emb)
        loss = F.mse_loss(pred * loss_mask.unsqueeze(-1), u * loss_mask.unsqueeze(-1), reduction="sum") / (
            torch.sum(loss_mask) * D
        )

        return loss
