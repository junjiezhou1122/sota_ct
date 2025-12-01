#!/usr/bin/env python3
"""
Shared ViT-style Transformer encoder over multi-scale tubelet embeddings.

This module mirrors the ViT blocks you provided (PreNorm/Attention/FFN),
adds a simple scale embedding (fine/mid/coarse), and is intended to load
Transformer weights from a pretrained 3D ViT while keeping tubelet/pos
embeddings learnable on your CT multi-scale setup.
"""

from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads),
            qkv,
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SharedViTEncoder(nn.Module):
    """
    Shared ViT-style Transformer over tubelet embeddings.

    For each scale:
        - add scale embedding (fine/mid/coarse)
        - run the same Transformer (weights can be loaded from pretrained ViT)
        - optional mask zero-out (padding)
    """

    def __init__(
        self,
        dim: int = 512,
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        # 0: fine, 1: mid, 2: coarse
        self.scale_embed = nn.Parameter(torch.randn(3, dim))

    def _encode_one_scale(
        self,
        x: torch.Tensor,   # (B, N, C)
        mask: torch.Tensor,  # (B, N)
        scale_id: int,
    ) -> torch.Tensor:
        # add scale embedding
        x = x + self.scale_embed[scale_id].unsqueeze(0).unsqueeze(0)
        # run transformer
        x = self.transformer(x)
        # zero-out padded tokens (mask: 1=valid, 0=pad)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x

    def forward(
        self,
        fine: torch.Tensor,
        mid: torch.Tensor,
        coarse: torch.Tensor,
        masks_fine: torch.Tensor,
        masks_mid: torch.Tensor,
        masks_coarse: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        F_fine = self._encode_one_scale(fine, masks_fine, scale_id=0)
        F_mid = self._encode_one_scale(mid, masks_mid, scale_id=1)
        F_coarse = self._encode_one_scale(coarse, masks_coarse, scale_id=2)
        return F_fine, F_mid, F_coarse


__all__ = ["SharedViTEncoder", "Transformer", "Attention", "FeedForward", "PreNorm"]
