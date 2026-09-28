import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dlm_an.model.conv import ResBlock1d
from dlm_an.model.dit.layers import ScalarEmbedder, FinalLinear
from dlm_an.model.dit.dit_encoder import DiTEncoder


class DurationPredictor(nn.Module):
    def __init__(self, predictor, log_scale):
        super().__init__()
        self.predictor = predictor
        self.log_scale = log_scale

    def compute_loss(self, x, x_mask, lin_d, return_stat=False):
        """
        Args:
            x (torch.tensor): batch of text representations.
                shape: (B, T, D)
            mask (torch.tensor): batch of masks.
                shape: (B, T)
            lin_d (torch.tensor): batch of ground-truth linear duration values.
                shape: (B, T)
        Returns:
            loss (torch.tensor): loss value.
                shape: (1)
            Optional:
                - mean (torch.tensor): predicted mean values.
                    shape: (B, T)
                - std (torch.tensor): predicted standard deviation values.
                    shape: (B, T)
                - z_score (torch.tensor): z-score values.
                    shape: (B, T)
        """
        x_mask = x_mask.unsqueeze(-1)  # (B, T, 1)
        lin_d = lin_d.unsqueeze(-1)  # (B, T, 1)

        o = self.predictor(x, x_mask)  # (B, T, 2)
        mean, log_std = torch.chunk(o, 2, dim=-1)
        std = torch.exp(log_std)
        var = std**2

        d = torch.log(lin_d + 1e-8) if self.log_scale else lin_d
        loss = torch.sum(
            F.gaussian_nll_loss(mean, d, var, reduction="none") * x_mask
        ) / torch.sum(x_mask)

        if return_stat:
            mean = mean.squeeze(-1).detach()
            std = std.squeeze(-1).detach()
            z_score = (d.squeeze(-1) - mean) / std
            return loss, mean, std, z_score
        else:
            return loss

    def forward(self, x, x_mask, temperature=0.0):
        """
        Args:
            x (torch.tensor): batch of text representations.
                shape: (B, T, D)
            mask (torch.tensor): batch of masks.
                shape: (B, T)
            temperature (float): temperature value for sampling.
        Returns:
            lin_d (torch.tensor): predicted linear duration values
                shape: (B, T)
        """
        x_mask = x_mask.unsqueeze(-1)  # (B, T, 1)
        o = self.predictor(x, x_mask) * x_mask  # (B, T, 2)
        mean, log_std = torch.chunk(o, 2, dim=-1)  # (B, T, 1), (B, T, 1)

        if temperature == 0.0:
            lin_d = torch.exp(mean - 1e-8) if self.log_scale else mean
            lin_d = self.round_duration(lin_d * x_mask)
            return lin_d.squeeze(-1)
        elif temperature > 0.0:
            std = torch.exp(log_std)
            lin_d = self.sample(
                mean.squeeze(-1), std.squeeze(-1), temperature=temperature
            )
            return lin_d * x_mask.squeeze(-1)
        else:
            raise ValueError("temperature should be non-negative.")

    def sample(self, mean, std, temperature=1.0):
        """
        Args:
            mean (torch.tensor): mean values.
                shape: (B, T)
            std (torch.tensor): standard deviation values.
                shape: (B, T)
        Returns:
            torch.tensor: sampled duration values.
                shape: (B, T)
        """
        d = mean + std * torch.randn(mean.size()).to(mean.device) * temperature
        lin_d = torch.exp(d - 1e-8) if self.log_scale else d
        return self.round_duration(lin_d)

    def round_duration(self, lin_d):
        """
        Args:
            lin_d (torch.tensor): batch of linear duration values.
                shape: (B, T, ...)
        Returns:
            torch.tensor: rounded duration values.
                shape: (B, T, ...)
        """
        return torch.round(torch.clamp(lin_d, min=1.0))

    @property
    def support_total_duration(self):
        return False


class FlowMatchingDurationPredictor(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        log_scale: bool,
        cfg_rate: float = 0.3,
    ):
        super().__init__()
        self.input_linear = nn.Linear(cond_dim + 1, embed_dim)
        self.dit = DiTEncoder(
            D=embed_dim,
            D_hidden=embed_dim * 4,
            N_head=num_heads,
            N_layer=num_layers,
            P_dropout=dropout_rate,
        )
        self.final_linear = FinalLinear(embed_dim, 1)

        self.t_encoder = ScalarEmbedder(256, embed_dim)
        self.average_duration_embedder = nn.Linear(1, embed_dim, bias=True)

        self.D = embed_dim
        self.log_scale = log_scale
        self.cfg_rate = cfg_rate

    def _forward(self, xt, p, t, cond, mask, avg_emb):
        """
        Args:
            xt (torch.tensor): noised values.
                shape: (B, T)
            p (torch.tensor): positions.
                shape: (B, T)
            t (torch.tensor): time steps.
                shape: (B)
            cond (torch.tensor): tokens representations.
                shape: (B, T, D_in)
            mask (torch.tensor): x's masks
                shape: (B, T)
            avg_emb (torch.tensor): embedded averaged values.
                shape: (B, T, D)
        Returns:
            u_hat (torch.tensor): predicted velocities.
                shape: (B, T)
        """
        B, T = xt.shape
        t_emb = self.t_encoder(t)  # (B, D)

        cond = torch.cat([cond, xt.unsqueeze(2)], dim=2)  # (B, T, D_in + 1)
        cond = self.input_linear(cond)  # (B, T, D)

        # TODO: switch to AdaLN conditioning
        cond = cond + avg_emb

        attn_mask = mask.unsqueeze(1).expand(-1, T, -1)  # (B, T, T)
        u_hat = self.dit(cond, p, t_emb, attn_mask)  # (B, T, D)
        u_hat = self.final_linear(u_hat, t_emb).squeeze(2) * mask  # (B, T)

        return u_hat

    def compute_loss(self, cond, mask, lin_d, t_scheduler="linear", sigma_min=1e-6):
        """
        Args:
            cond (torch.tensor): batch of text representations.
                shape: (B, T, D_in)
            mask (torch.tensor): batch of masks.
                shape: (B, T)
            lin_d (torch.tensor): batch of ground-truth linear duration values.
                shape: (B, T)
            t_scheduler (str): time schedule for flow matching training
        Returns:
            loss (torch.tensor): loss value.
                shape: (1)
        """
        B, T, D_in = cond.shape
        x_lengths = mask.sum(dim=1)  # (B,)

        d = torch.log(lin_d + 1e-8) if self.log_scale else lin_d

        avg_lin_d = (lin_d * mask).sum(dim=1) / x_lengths  # (B,)
        avg_d = torch.log(avg_lin_d + 1e-8) if self.log_scale else avg_lin_d  # (B,)
        avg_d_emb = self.average_duration_embedder(avg_d.unsqueeze(1))  # (B, D)
        avg_d_emb = avg_d_emb.unsqueeze(1).repeat(1, T, 1)  # (B, T, D)

        # NOTE: CFG is only applied to the mean value, not to the linguistic condition
        if self.cfg_rate > 0.0:
            cfg_mask = torch.rand(B, device=cond.device) > self.cfg_rate
            avg_d_emb = avg_d_emb * cfg_mask.view(-1, 1, 1)  # (B, T, D)

        t = torch.rand([B], device=cond.device, dtype=cond.dtype)
        if t_scheduler == "cosine":
            t = 1 - torch.cos(t * 0.5 * math.pi)
        _t = t.view(-1, 1)
        z = torch.randn_like(d)

        xt = (1 - (1 - sigma_min) * _t) * z + _t * d
        u = d - (1 - sigma_min) * z

        _p = torch.arange(0, T, 1, dtype=cond.dtype, device=cond.device)  # (T,)
        p = _p.unsqueeze(0).repeat(B, 1)  # (B, T)

        u_hat = self._forward(xt, p, t, cond, mask, avg_d_emb)  # (B, T)

        loss = F.mse_loss(u_hat * mask, u * mask, reduction="sum") / torch.sum(
            mask
        )  # (1,)

        return loss

    def forward(
        self,
        cond,
        mask,
        total_duration=None,
        n_timesteps=10,
        t_scheduler="linear",
        cfg_scale=0.1,
    ):
        """
        Args:
            cond (torch.tensor): batch of text representations.
                shape: (B, T, D_in)
            mask (torch.tensor): batch of masks.
                shape: (B, T)
            Optional:
                - total_duration: total duration to approach
                    shape: (B)
                - n_timesteps: number of sampling steps
                - t_scheduler: time scheduler
                - cfg_scale: CFG scale for total duration conditioning
        Returns:
            lin_d (torch.tensor): predicted linear duration values
                shape: (B, T)
        """
        B, T, D_in = cond.shape
        x_lengths = mask.sum(dim=1)

        if total_duration is not None:
            avg_lin_d = total_duration / x_lengths
            avg_d = torch.log(avg_lin_d + 1e-8) if self.log_scale else avg_lin_d  # (B,)
            avg_d_emb = self.average_duration_embedder(avg_d.unsqueeze(1))  # (B, D)
            avg_d_emb = avg_d_emb.unsqueeze(1).repeat(1, T, 1)  # (B, T, D)
        else:
            avg_d_emb = torch.zeros((B, T, self.D), device=cond.device)
            # Disable CFG when total durations are not given
            cfg_scale = 0.0

        t_span = torch.linspace(
            0, 1, n_timesteps + 1, device=cond.device, dtype=cond.dtype
        )
        if t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        _p = torch.arange(0, T, 1, dtype=cond.dtype, device=cond.device)  # (T,)
        p = _p.unsqueeze(0).repeat(B, 1)  # (B, T)

        d = torch.randn((B, T), device=cond.device)

        # ODE integration
        t, dt = t_span[0].unsqueeze(dim=0), t_span[1] - t_span[0]
        for step in range(1, len(t_span)):
            dphi_dt = self._forward(d, p, t, cond, mask, avg_d_emb)
            if cfg_scale != 0.0:
                cfg_dphi_dt = self._forward(
                    d, p, t, cond, mask, torch.zeros_like(avg_d_emb)
                )
            else:
                cfg_dphi_dt = torch.zeros_like(dphi_dt)
            dphi_dt = (1 + cfg_scale) * dphi_dt - cfg_scale * cfg_dphi_dt
            d = d + dphi_dt * dt

            # Update time
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        lin_d = torch.exp(d) if self.log_scale else d  # (B, T)

        return self.round_duration(lin_d, total_duration) * mask

    def round_duration(self, lin_d, total_duration=None):
        """
        Args:
            lin_d (torch.tensor): batch of linear duration values.
                shape: (B, T, ...)
        Returns:
            torch.tensor: rounded duration values.
                shape: (B, T, ...)
        """
        # NOTE: the flow-matching model is trained with rounded duration targets,
        # so the generated values are naturally around integers.
        rounded_d = torch.round(torch.clamp(lin_d, min=1.0))
        if total_duration is not None:
            rounded_d = scale_to_total_duration(rounded_d, total_duration)
        return rounded_d

    @property
    def support_total_duration(self):
        return True


def scale_to_total_duration(d, total_duration):
    """
    Args:
        d (torch.tensor): batch of duration values.
            shape: (B, T)
        total_duration (torch.tensor): total duration values for each sample.
            shape: (B)
    Returns:
        rounded_d (torch.tensor): scaled duration values that sum up to the given total duration.
            shape: (B, T)
    """
    # Sum the durations along the time dimension
    sum_d = torch.sum(d, dim=-1, keepdim=True)  # shape: (B, 1)

    # Avoid division by zero by replacing zeros with ones
    sum_d[sum_d == 0] = 1.0

    # Scale the durations to match the total duration
    scaled_d = d * (total_duration.view(-1, 1) / sum_d)  # shape: (B, T)

    # Round the scaled durations to the nearest integers
    rounded_d = torch.round(scaled_d)

    # Calculate the difference between the total duration and the sum of the rounded durations
    diff = total_duration.view(-1) - torch.sum(rounded_d, dim=-1)
    diff = diff.long()

    # Adjust the rounded durations to ensure the sum matches the total duration
    for i in range(diff.size(0)):
        if diff[i] > 0:
            # If the sum of rounded durations is less than the total duration, add 1 to some elements
            indices = torch.argsort(scaled_d[i] - rounded_d[i], descending=True)
            rounded_d[i, indices[: diff[i]]] += 1
        elif diff[i] < 0:
            # If the sum of rounded durations is more than the total duration, subtract 1 from some elements
            indices = torch.argsort(rounded_d[i] - scaled_d[i], descending=True)
            rounded_d[i, indices[: abs(diff[i])]] -= 1

    return rounded_d
