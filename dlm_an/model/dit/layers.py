import torch
from torch import nn, Tensor
from torch.nn.functional import layer_norm, gelu

from .utils import encode_scalar, modulate


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, D: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))
        self.bias = nn.Parameter(torch.zeros(D)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        """[..., D] -> [..., D]"""
        return layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class FeedForwardModule(nn.Module):
    def __init__(
        self,
        D: int,
        D_hidden: int,
        P_dropout: float,
        bias: bool = True,
    ):
        """
        Args:
            D (int): Input feature dimension.
            D_hidden (int): Hidden unit dimension.
            P_dropout (float): dropout value for first Linear Layer.
            bias (bool): If linear layers should have bias.
            d_cond (int, optional): The channels of conditional tensor.
        """
        super().__init__()
        self.w_1 = nn.Linear(D, D_hidden, bias=bias)
        self.drop_1 = nn.Dropout(P_dropout)
        self.w_2 = nn.Linear(D_hidden, D, bias=bias)
        self.drop_2 = nn.Dropout(P_dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): [..., D].
        Returns:
            y (Tensor): [..., D].
        """
        x = self.w_1(x)
        try:
            x = gelu(x, approximate="tanh")
        except TypeError:
            x = gelu(x)
        x = self.drop_1(x)
        x = self.w_2(x)
        return self.drop_2(x)


class ScalarEmbedder(nn.Module):
    def __init__(self, F: int, D: int):
        """Encode a scalar into a vector.
        Args:
            F (int): The frequency embedding size.
            D (int): The embedding size.
        """
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(F, D, bias=True), nn.SiLU(), nn.Linear(D, D, bias=True))
        self.F = F

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t (Tensor): [N].
        Returns:
            p (Tensor): [N, D].
        """
        t_code = encode_scalar(t, self.F)
        t_emb = self.mlp.forward(t_code)
        return t_emb


class FinalLinear(nn.Module):
    def __init__(self, D, D_out):
        super().__init__()
        self.norm_final = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(D, D_out, bias=True)
        self.AdaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(D, 2 * D, bias=True))

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """DiT forward function.
        Args:
            x (Tensor): [N, T, D].
            c (Tensor): [N, D].
        Returns:
            y (Tensor): [N, T, D_out]
        """
        shift, scale = self.AdaLN_modulation.forward(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear.forward(x)
        return x
