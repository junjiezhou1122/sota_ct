#!/usr/bin/env python3
"""
End-to-end shared vision encoder for multi-scale tubelets.

This "shared-only" version uses:
    - a shared TubeletEmbedding3D
    - a shared Transformer encoder (SharedViTEncoder)
to produce multi-scale vision tokens.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .tubelet_embedding import TubeletEmbedding3D
from .shared_transformer import SharedViTEncoder


class VisionEncoder(nn.Module):
    """
    Shared vision encoder for multi-scale tubelets.

    Inputs:
        tubelets_fine:   (B, Nf, Df, Hf, Wf)
        tubelets_mid:    (B, Nm, Dm, Hm, Wm)
        tubelets_coarse: (B, Nc, Dc, Hc, Wc)
        masks_fine:      (B, Nf) 1 for valid, 0 for padded
        masks_mid:       (B, Nm)
        masks_coarse:    (B, Nc)
    Outputs:
        F_fine:   (B, Nf, C)
        F_mid:    (B, Nm, C)
        F_coarse: (B, Nc, C)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = TubeletEmbedding3D(embed_dim=embed_dim)
        self.encoder = SharedViTEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        tubelets_fine: torch.Tensor,
        tubelets_mid: torch.Tensor,
        tubelets_coarse: torch.Tensor,
        masks_fine: torch.Tensor,
        masks_mid: torch.Tensor,
        masks_coarse: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Efine = self.embed(tubelets_fine)
        Emid = self.embed(tubelets_mid)
        Ecoarse = self.embed(tubelets_coarse)

        F_fine, F_mid, F_coarse = self.encoder(
            Efine,
            Emid,
            Ecoarse,
            masks_fine,
            masks_mid,
            masks_coarse,
        )

        return F_fine, F_mid, F_coarse


__all__ = ["VisionEncoder"]

