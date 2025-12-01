#!/usr/bin/env python3
"""
Tubelet embedding for 3D CT volumes.

This module defines `TubeletEmbedding3D`, which converts each 3D
tubelet (D, H, W) into a C-dimensional feature vector using a small
3D CNN followed by global average pooling.
"""

from typing import Tuple

import torch
import torch.nn as nn


class TubeletEmbedding3D(nn.Module):
    """
    Shared tubelet embedding: 3D tubelet -> C-dim vector.

    Expected input per scale:
        tubelets: (B, N, D, H, W)
    Output:
        embeddings: (B, N, C)
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1),
            nn.InstanceNorm3d(embed_dim),
            nn.GELU(),
        )

    def forward(self, tubelets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tubelets: (B, N, D, H, W)
        Returns:
            (B, N, C) where C = embed_dim
        """
        B, N, D, H, W = tubelets.shape

        # Add channel dimension and merge batch + tubelet dims
        x = tubelets.view(B * N, 1, D, H, W)
        x = self.conv(x)  # (B*N, C, D, H, W)

        # Global average pooling over spatial dims
        x = x.mean(dim=(2, 3, 4))  # (B*N, C)

        # Restore (B, N, C)
        x = x.view(B, N, -1)
        return x


__all__ = ["TubeletEmbedding3D"]

