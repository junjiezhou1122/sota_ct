#!/usr/bin/env python3
"""
Shared vision encoder for multi-scale tubelets.

Design:
    - Learned tubelet embedding (3D Conv + pooling) -> (B, N, C)
    - Scale embedding (fine / mid / coarse)
    - Shared ViT-style Transformer (can load pretrained weights)
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
        depth: int = 4,
        heads: int = 8,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        pretrained_vit: nn.Module | None = None,
    ):
        super().__init__()
        self.embed = TubeletEmbedding3D(embed_dim=embed_dim)
        self.encoder = SharedViTEncoder(
            dim=embed_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        # Optionally load transformer weights from a pretrained ViT
        if pretrained_vit is not None and hasattr(pretrained_vit, "transformer"):
            self.encoder.transformer.load_state_dict(
                pretrained_vit.transformer.state_dict(), strict=True
            )
            print("Loaded transformer weights from pretrained ViT.")

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
