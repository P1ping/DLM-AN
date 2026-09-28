# Copyright (c) 2024 Zhijun Liu (zhijunliu1@link.cuhk.edu.cn)
#               2025 Qibing Bai (qibingbai@link.cuhk.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from math import sqrt
from typing import Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor, BoolTensor
from torch.nn.functional import dropout


def compute_r(p: Tensor, C: int, theta: float = 10000.0) -> Tensor:
    """Compute complex rotation from float positions.
    Args:
        p (Tensor): [N, T], tensor denoting position in each sequence.
        C (int): the dimension to rotate.
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
    Returns:
        r (Tensor): [N, T, C // 2].
    """
    s = torch.arange(0, C, 2, dtype=torch.float32, device=p.device)[: (C // 2)] / C  # [C // 2]
    m = 1.0 / (theta**s)  # [C // 2]
    diff_dim = len(p.shape) - 1
    r = m.view(*(1,) * diff_dim, -1) * p.float().unsqueeze(-1)  # [N, T, C // 2]
    return r


def rotate(x: Tensor, r: Tensor) -> Tensor:
    """For different heads, the same rotation is applied.
    Args:
        x (Tensor): [N, H, T, C], where C is even and to be rotated in pairs.
        r (Tensor): [N, T, C // 2].
    Returns:
        x (Tensor): [N, H, T, C], the rotated tensor.
    """
    C = x.shape[-1]

    x_reshape = x.float().reshape(*x.shape[:-1], C // 2, 2)
    # [N, H, T, C // 2, 2]

    r = r.unsqueeze(-3)

    return (
        torch.stack(
            [
                x_reshape[..., 0] * r.cos() - x_reshape[..., 1] * r.sin(),
                x_reshape[..., 0] * r.sin() + x_reshape[..., 1] * r.cos(),
            ],
            dim=-1,
        )
        .flatten(-2, -1)
        .type_as(x)
    )


class RoPESelfAttention(nn.Module):
    def __init__(
        self,
        D: int,
        N_head: int,
        bias: bool = False,
        P_dropout: float = 0,
        is_causal: bool = False,
    ) -> None:
        super().__init__()
        assert D % N_head == 0
        self.H = N_head
        self.D = D
        self.C = D // N_head
        self.P_dropout = P_dropout

        self.linear_qkv = nn.Linear(D, 3 * D, bias=bias)
        self.linear_out = nn.Linear(D, D, bias=bias)
        self.last_drop = nn.Dropout(P_dropout)

        self.is_causal = is_causal

    def compute_qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute key, queue, and value from input tensor x.
        Args:
            x (Tensor): [N, T, D].
        Returns:
            q, k, v (Tensor): [N, H, T, C], C = D // H.
        """
        H, D, C = self.H, self.D, self.C
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.linear_qkv(x).split(D, dim=-1)  # [N, T, D] x 3
        try:
            q = torch.unflatten(q, -1, (H, C)).transpose(-2, -3)
            k = torch.unflatten(k, -1, (H, C)).transpose(-2, -3)
            v = torch.unflatten(v, -1, (H, C)).transpose(-2, -3)
        except AttributeError:
            N, T, _ = x.shape
            q = q.reshape(N, T, H, C).transpose(-2, -3)
            k = k.reshape(N, T, H, C).transpose(-2, -3)
            v = v.reshape(N, T, H, C).transpose(-2, -3)
        # [N, H, T, C]
        return q, k, v

    def forward(
        self,
        x: Tensor,
        r: Tensor,
        mask: BoolTensor,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute self-attention.
        Args:
            x (Tensor): [N, T, D]. T can be T_total (no cache) or T_new (with cache).
            r (Tensor): [N, T_total, C // 2], where C = D // H.
            mask (BoolTensor): [N, T_total, T_total], see document on SDPA in PyTorch.
            k_cache (Optional[Tensor]): [N, H, T_cache, C], cached keys.
            v_cache (Optional[Tensor]): [N, H, T_cache, C], cached values.
        Returns:
            y (Tensor): [N, T, D].
            k_full, v_full (Tensor): [N, H, T_total, C].
        """
        mask_h = mask.unsqueeze(-3).expand(-1, self.H, -1, -1)  # [N, H, T_total, T_total]
        q, k_new, v_new = self.compute_qkv(x)  # [N, H, T, C]

        # Handle KV cache
        if k_cache is not None and v_cache is not None:
            k_full = torch.cat([k_cache, k_new], dim=-2)  # [N, H, T_total, C]
            v_full = torch.cat([v_cache, v_new], dim=-2)  # [N, H, T_total, C]
            r_q = r[:, -q.shape[-2] :, :]  # [N, T, C // 2]
            mask_h = mask_h[:, :, -q.shape[-2] :, :]  # [N, H, T, T_total]
        else:
            k_full, v_full = k_new, v_new
            r_q = r  # [N, T_total, C // 2]
            if k_cache is not None or v_cache is not None:
                warnings.warn("KV cache should be both provided or not provided. KV cache is not used.")
        assert k_full.shape[-2] == mask.shape[-1], "Mask shape must match key shape."
        assert v_full.shape[-2] == mask.shape[-1], "Mask shape must match value shape."

        q_rotated = rotate(q, r_q)  # [N, H, T, C]
        k_rotated = rotate(k_full, r)

        using_cache = k_cache is not None and v_cache is not None
        # When to use SDPA: 1) causal attention & 2) not using cache & 3) no left padding
        if self.is_causal and (not using_cache) and torch.all(mask[:, -1]).item():
            attn_mask = None
            is_causal = True
        else:
            attn_mask = mask_h
            is_causal = False

        try:
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                o = F.scaled_dot_product_attention(
                    q_rotated,
                    k_rotated,
                    v_full,
                    dropout_p=self.P_dropout if self.training else 0,
                    attn_mask=attn_mask,
                    is_causal=is_causal,
                )  # [N, H, T, C]
        except:
            w = q_rotated @ k_rotated.transpose(-1, -2) / sqrt(self.C)  # [N, ..., H, T, T_total]
            w = w.masked_fill(~mask_h, float("-inf"))
            all_masked = (w == float("-inf")).all(dim=-1, keepdim=True)  # Avoid NaN in padded regions
            attn = torch.softmax(w, dim=-1)
            attn = attn.masked_fill(all_masked, 0.0)
            if self.training:
                attn = dropout(attn, p=self.P_dropout)
            o = attn @ v_full
            o = o.masked_fill(all_masked, 0.0)

        y = o.transpose(-2, -3).contiguous().flatten(-2, -1)  # [N, T, D]
        y = self.linear_out.forward(y)
        y = self.last_drop.forward(y)
        return y, k_full, v_full

    def get_attn(self, x: Tensor, r: Tensor, mask: BoolTensor) -> Tensor:
        """Compute the attention weights.
        Args: Same as forward(...)
        Returns:
            attn (Tensor): [N, H, T, T].
        """
        with torch.no_grad():
            q, k, _ = self.compute_qkv(x)  # [N, ..., H, T, C]
            q = rotate(q, r)
            k = rotate(k, r)
            w = q @ k.transpose(-1, -2) / sqrt(self.C)  # [N, ..., H, T, T]
            w = w.masked_fill(~mask, float("-inf"))
            return torch.softmax(w, dim=-1)


class RoPECrossAttention(nn.Module):
    def __init__(
        self,
        D: int,
        N_head: int,
        bias: bool = False,
        P_dropout: float = 0,
    ) -> None:
        super().__init__()
        assert D % N_head == 0
        self.H = N_head
        self.D = D
        self.C = D // N_head
        self.P_dropout = P_dropout

        # Separate projections for query vs key/value
        self.linear_q = nn.Linear(D, D, bias=bias)
        self.linear_kv = nn.Linear(D, 2 * D, bias=bias)
        self.linear_out = nn.Linear(D, D, bias=bias)
        self.last_drop = nn.Dropout(P_dropout)

    def compute_q(self, x: Tensor) -> Tensor:
        """Compute query from target sequence tensor x.
        Args:
            x (Tensor): [N, T_q, D].
        Returns:
            q (Tensor): [N, H, T_q, C], C = D // H.
        """
        H, D, C = self.H, self.D, self.C
        q = self.linear_q(x)  # [N, T_q, D]
        try:
            q = torch.unflatten(q, -1, (H, C)).transpose(-2, -3)
        except AttributeError:
            N, T, _ = x.shape
            q = q.reshape(N, T, H, C).transpose(-2, -3)
        # [N, H, T_q, C]
        return q

    def compute_kv(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute key and value from source sequence tensor y.
        Args:
            y (Tensor): [N, T_kv, D].
        Returns:
            k, v (Tensor): [N, H, T_kv, C], C = D // H.
        """
        H, D, C = self.H, self.D, self.C
        N, T_kv, _ = y.shape
        if T_kv > 0:
            k, v = self.linear_kv(y).split(D, dim=-1)  # [N, T_kv, D] x 2
            k = k.reshape(y.shape[0], y.shape[1], H, C).transpose(-2, -3)  # [N, H, T_kv, C]
            v = v.reshape(y.shape[0], y.shape[1], H, C).transpose(-2, -3)  # [N, H, T_kv, C]
        else:
            k = torch.empty(N, H, 0, C, device=y.device, dtype=y.dtype)
            v = torch.empty(N, H, 0, C, device=y.device, dtype=y.dtype)
        return k, v

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        r_q: Tensor,
        r_kv: Tensor,
        mask: BoolTensor,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute cross-attention.
        Args:
            x (Tensor): [N, T_q, D], query sequence.
            y (Tensor): [N, T_kv, D], key/value sequence. T_kv can be T_kv_total (no cache) or T_kv_new (with cache).
            r_q (Tensor): [N, T_q, C // 2], rotation for query sequence.
            r_kv (Tensor): [N, T_kv_total, C // 2], rotation for key/value sequence.
            mask (BoolTensor): [N, T_q, T_kv_total], cross-attention mask.
            k_cache (Optional[Tensor]): [N, H, T_kv_cache, C], cached keys.
            v_cache (Optional[Tensor]): [N, H, T_kv_cache, C], cached values.
        Returns:
            output (Tensor): [N, T_q, D], attention output.
            k_full, v_full (Tensor): [N, H, T_kv_total, C], keys and values.
        """
        # Expand mask for multi-head attention
        mask_h = mask.unsqueeze(-3).expand(-1, self.H, -1, -1)  # [N, H, T_q, T_kv]

        # Compute queries from target sequence
        q = self.compute_q(x)  # [N, H, T_q, C]

        # Compute keys and values from source sequence
        k_new, v_new = self.compute_kv(y)  # [N, H, T_kv_new, C]

        # Handle KV cache
        if k_cache is not None and v_cache is not None:
            k_full = torch.cat([k_cache, k_new], dim=-2)  # [N, H, T_kv_total, C]
            v_full = torch.cat([v_cache, v_new], dim=-2)  # [N, H, T_kv_total, C]
            r_q = r_q[:, -q.shape[-2] :, :]  # [N, T_q, C // 2]
            mask_h = mask_h[:, :, -q.shape[-2] :, :]  # [N, H, T_q, T_kv_total]
        else:
            k_full, v_full = k_new, v_new
            if k_cache is not None or v_cache is not None:
                warnings.warn("KV cache should be both provided or not provided. KV cache is not used.")

        # Apply rotary position embeddings
        q = rotate(q, r_q)
        k = rotate(k_full, r_kv)

        # Compute attention
        try:
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                o = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v_full,
                    dropout_p=self.P_dropout if self.training else 0,
                    attn_mask=mask_h,
                )  # [N, H, T_q, C]
        except:
            # Fall back to manual implementation
            w = q @ k.transpose(-1, -2) / sqrt(self.C)  # [N, H, T_q, T_kv]
            w = w.masked_fill(~mask_h, float("-inf"))
            all_masked = (w == float("-inf")).all(dim=-1, keepdim=True)  # Avoid NaN in padded regions
            attn = torch.softmax(w, dim=-1)
            attn = attn.masked_fill(all_masked, 0.0)
            if self.training:
                attn = dropout(attn, p=self.P_dropout)
            o = attn @ v_full  # [N, H, T_q, C]
            o = o.masked_fill(all_masked, 0.0)

        # Reshape output and project
        y = o.transpose(-2, -3).contiguous().flatten(-2, -1)  # [N, T_q, D]
        y = self.linear_out(y)
        y = self.last_drop(y)

        return y, k_full, v_full

    def get_attn(
        self,
        x: Tensor,
        y: Tensor,
        r_q: Tensor,
        r_kv: Tensor,
        mask: BoolTensor,
    ) -> Tensor:
        """Compute the cross-attention weights.
        Args: Same as forward(...)
        Returns:
            attn (Tensor): [N, H, T_q, T_kv], attention weights.
        """
        with torch.no_grad():
            q = self.compute_q(x)  # [N, H, T_q, C]
            k, _ = self.compute_kv(y)  # [N, H, T_kv, C]

            q = rotate(q, r_q)
            k = rotate(k, r_kv)

            w = q @ k.transpose(-1, -2) / sqrt(self.C)  # [N, H, T_q, T_kv]
            w = w.masked_fill(~mask.unsqueeze(-3), float("-inf"))
            return torch.softmax(w, dim=-1)
