from typing import Optional, List, Tuple

import torch
from torch import nn, Tensor, BoolTensor

from dlm_an.model.rope import compute_r, RoPESelfAttention
from dlm_an.model.dit.layers import FeedForwardModule


class RotaryEncoderLayer(nn.Module):
    def __init__(
        self,
        D: int,
        D_hidden: int,
        N_head: int,
        P_dropout: float,
        is_causal: bool = False,
    ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(D)
        self.attn = RoPESelfAttention(
            D, N_head, P_dropout=P_dropout, is_causal=is_causal
        )
        self.ffn_norm = nn.LayerNorm(D)
        self.ffn = FeedForwardModule(D, D_hidden, P_dropout, bias=True)

    def forward(
        self,
        x: Tensor,
        r: Tensor,
        mask: BoolTensor,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x (Tensor): [N, ..., T, D].
            r (Tensor): [N, ..., T, C//2], rotation in RoPE.
            mask (BoolTensor): [N, T, T], attention mask, True for valid positions.
            k_cache (Tensor): [N, ..., H, T_cache, D], cached keys.
            v_cache (Tensor): [N, ..., H, T_cache, D], cached values.
        Returns:
            y (Tensor): [N, ..., T, D].
        """
        x1 = self.attn_norm.forward(x)
        x2, k_cache, v_cache = self.attn.forward(x1, r, mask, k_cache, v_cache)
        x = x + x2
        x3 = self.ffn_norm.forward(x)
        x4 = self.ffn.forward(x3)
        x = x + x4
        return x, k_cache, v_cache


class RotaryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        is_causal: bool = False,
    ):
        super().__init__()
        self.input_proj = (
            nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else None
        )
        self.blocks = nn.ModuleList(
            [
                RotaryEncoderLayer(
                    embed_dim, hidden_dim, num_heads, dropout_rate, is_causal
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.C = embed_dim // num_heads

    def forward(
        self,
        x: Tensor,
        p: Tensor,
        mask: BoolTensor,
        return_hidden: bool = False,
        stop_grad_layer: int = None,
    ) -> Tensor:
        """
        Args:
            x (Tensor): [N, ..., T, D].
            p (Tensor): [N, ..., T], the position tensor.
            t_emb (Tensor): [N, ...], the time embedding tensor.
            mask (BoolTensor): [N, T, T], attention mask.
            return_hidden (bool): whether to return hidden states of all layers.
            stop_grad_layer (Optional[int]): if specified, stop gradient after this layer.
        Returns:
            y (Tensor): [N, ..., T, D].
        """
        r = compute_r(p, self.C)

        if self.input_proj is not None:
            x = self.input_proj(x)

        hidden_states = []
        for idx, block in enumerate(self.blocks):
            x, _, _ = block.forward(x, r, mask)
            if return_hidden:
                hidden_states.append(x)
            if stop_grad_layer is not None and idx == stop_grad_layer:
                x = x.detach()
        x = self.final_norm(x)

        if return_hidden:
            return x, hidden_states
        return x

    def forward_incremental(
        self,
        x: Tensor,
        p: Tensor,
        mask: BoolTensor,
        k_caches: List[Tensor],
        v_caches: List[Tensor],
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Args:
            x (Tensor): [N, ..., T, D].
            p (Tensor): [N, ..., T], the position tensor.
            mask (BoolTensor): [N, T, T], attention mask.
            k_caches (List[Tensor]): list of cached keys for each layer.
            v_caches (List[Tensor]): list of cached values for each layer.
        Returns:
            y (Tensor): [N, ..., T, D].
            new_k_caches (List[Tensor]): updated list of cached keys for each layer.
            new_v_caches (List[Tensor]): updated list of cached values for each layer.
        """
        r = compute_r(p, self.C)

        if self.input_proj is not None:
            x = self.input_proj(x)

        new_k_caches = []
        new_v_caches = []
        for i, block in enumerate(self.blocks):
            x, new_k_cache, new_v_cache = block.forward(
                x,
                r,
                mask,
                k_cache=k_caches[i],
                v_cache=v_caches[i],
            )
            new_k_caches.append(new_k_cache)
            new_v_caches.append(new_v_cache)
        x = self.final_norm(x)
        return x, new_k_caches, new_v_caches

    def output_size(self):
        return self.embed_dim
