import math
import torch
from torch import Tensor


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """
    Args:
        x (Tensor): [N, ..., T, D].
        shift (Tensor): [N, ..., D].
        scale-1.0 (Tensor): [N, ..., D].
    """
    return x * (1 + scale.unsqueeze(-2)) + shift.unsqueeze(-2)


def encode_scalar(scalar: Tensor, D: int, c: float = 0.1) -> Tensor:
    """Encode scalar into sinusoidal vector.
    Args:
        scalar (Tensor): [B].
        c (float): a constant controlling the embeddings.
    Returns:
        code (Tensor): [B, D].
    """
    D_half = D // 2
    frequencies = torch.exp(-math.log(c) * torch.arange(start=0, end=D_half, dtype=torch.float32) / D_half).to(
        device=scalar.device
    )
    args = scalar[:, None].float() * frequencies[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
