import torch
from torch import nn, Tensor, BoolTensor

from dlm_an.model.rope import compute_r, RoPESelfAttention
from .layers import LayerNorm, FeedForwardModule
from .utils import modulate


class DiTEncoderLayer(nn.Module):
    def __init__(self, D: int, D_hidden: int, N_head: int, P_dropout: float):
        super().__init__()
        self.attn_norm = LayerNorm(D)
        self.attn = RoPESelfAttention(D, N_head, P_dropout=P_dropout)
        self.ffn_norm = LayerNorm(D)
        self.ffn = FeedForwardModule(D, D_hidden, P_dropout, bias=True)
        self.AdaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(D, 6 * D, bias=True))

    def forward(self, x: Tensor, c: Tensor, r: Tensor, mask: BoolTensor) -> Tensor:
        """
        Args:
            x (Tensor): [N, ..., T, D].
            c (Tensor): [N, ..., D], conditioning vector for AdaLN.
            r (Tensor): [N, ..., T, C//2], rotation in RoPE.
            mask (BoolTensor): [N, T, T], attention mask, True for valid positions.
        Returns:
            y (Tensor): [N, ..., T, D].
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.AdaLN_modulation(c).chunk(6, dim=-1)
        # [N, ..., D]
        x1 = self.attn_norm.forward(x)
        x2 = modulate(x1, shift_msa, scale_msa)
        x3, _, _ = self.attn.forward(x2, r, mask)
        x4 = gate_msa.unsqueeze(-2) * x3
        x = x + x4
        x5 = self.ffn_norm.forward(x)
        x6 = modulate(x5, shift_mlp, scale_mlp)
        x7 = self.ffn.forward(x6)
        x8 = gate_mlp.unsqueeze(-2) * x7
        x = x + x8
        return x

    def get_attn(self, x: Tensor, c: Tensor, r: Tensor, mask: BoolTensor) -> Tensor:
        """A hack to obtain the attention matrix.
        Args:
            x (Tensor): [N, ..., T, D].
            r (ComplexFloatTensor): [N, ..., T, C // 2].
            mask (BoolTensor): [N, T, T], attention mask.
        Returns:
            attn (Tensor): [N, ..., H, T, T] the attention weights.
        """
        state = self.training
        with torch.no_grad():
            self.eval()
            shift_msa, scale_msa, _, _, _, _ = self.AdaLN_modulation.forward(c).chunk(
                6, dim=-1
            )
            x1 = self.attn_norm.forward(x)
            x2 = modulate(x1, shift_msa, scale_msa)
            attn = self.attn.get_attn(x2, r, mask)
        self.train(state)
        return attn


class DiTEncoder(nn.Module):
    def __init__(
        self,
        D: int,
        D_hidden: int,
        N_head: int,
        N_layer: int,
        P_dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [DiTEncoderLayer(D, D_hidden, N_head, P_dropout) for _ in range(N_layer)]
        )
        self.C = D // N_head

    def forward(self, x: Tensor, p: Tensor, t_emb: Tensor, mask: BoolTensor) -> Tensor:
        """
        Args:
            x (Tensor): [N, ..., T, D].
            p (Tensor): [N, ..., T], the position tensor.
            t_emb (Tensor): [N, ...], the time embedding tensor.
            mask (BoolTensor): [N, T, T], attention mask.
        Returns:
            y (Tensor): [N, ..., T, D].
        """
        r = compute_r(p, self.C)
        for block in self.blocks:
            x = block.forward(x, t_emb, r, mask)
        return x

    def get_attn(self, x: Tensor, p: Tensor, t_emb: Tensor, mask: BoolTensor) -> Tensor:
        """Compute attention matrix.
        Args:
            x (Tensor): [N, ..., T, D].
            p (Tensor): [N, ..., T], the position tensor.
            t_emb (Tensor): [N, ...], the time embedding tensor.
            mask (BoolTensor): [N, T, T], attention mask.
        Returns:
            attn (Tensor): [N, ..., N_layer, H, T, T].
        """
        state = self.training
        attns = []
        with torch.inference_mode():
            self.eval()
            r = compute_r(p, self.C)
            for block in self.blocks:
                attn = block.get_attn(x, t_emb, r, mask)
                x = block.forward(x, t_emb, r, mask)
                attns.append(attn)
        self.train(state)
        return torch.stack(attns, dim=-4)
