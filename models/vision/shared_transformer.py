#!/usr/bin/env python3
"""
Shared Transformer encoder over multi-scale tubelet embeddings.

The `SharedViTEncoder` processes fine/mid/coarse tubelet embeddings
using a single TransformerEncoder with learned scale embeddings.
"""

from typing import Tuple

import torch
import torch.nn as nn


class SharedViTEncoder(nn.Module):
    """
    Shared Transformer encoder over tubelet embeddings.

    For each scale, we:
        - add a learned scale embedding
        - apply the same TransformerEncoder (shared weights)
        - use key_padding_mask from the corresponding masks
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 0: fine, 1: mid, 2: coarse
        self.scale_embed = nn.Parameter(torch.randn(3, embed_dim))

    def _encode_one_scale(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        scale_id: int,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, N, C) embeddings
            mask: (B, N) 1 for valid, 0 for padded
        Returns:
            encoded: (B, N, C)
        """
        # Ensure mask is boolean for key_padding_mask
        if mask.dtype != torch.bool:
            key_padding_mask = mask == 0
        else:
            key_padding_mask = ~mask  # True = pad

        # Add scale embedding
        x = x + self.scale_embed[scale_id].unsqueeze(0).unsqueeze(0)

        # key_padding_mask: (B, N), True at PAD positions
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
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
        """
        Args:
            fine, mid, coarse: (B, N*, C)
            masks_*: (B, N*) 1 for valid, 0 for padded
        Returns:
            F_fine, F_mid, F_coarse: (B, N*, C)
        """
        F_fine = self._encode_one_scale(fine, masks_fine, scale_id=0)
        F_mid = self._encode_one_scale(mid, masks_mid, scale_id=1)
        F_coarse = self._encode_one_scale(coarse, masks_coarse, scale_id=2)
        return F_fine, F_mid, F_coarse


__all__ = ["SharedViTEncoder"]

