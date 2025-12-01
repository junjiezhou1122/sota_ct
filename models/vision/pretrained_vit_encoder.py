#!/usr/bin/env python3
"""
Vision encoder wrapper for a pretrained 3D ViT backbone.

`PretrainedViTVisionEncoder` treats each tubelet as a small 3D volume
and runs all tubelets (fine/mid/coarse) through the same pretrained
ViT3D model (shared weights), returning one vector per tubelet.
"""

from typing import Tuple

import torch
import torch.nn as nn


class PretrainedViTVisionEncoder(nn.Module):
    """
    Vision encoder that wraps a pretrained 3D ViT backbone.

    The idea:
        - Treat each tubelet (D, H, W) as a small 3D volume.
        - Reshape tubelets into the shape expected by a pretrained
          ViT3D (B*N, C, H, W, D).
        - Run all tubelets through the SAME pretrained backbone
          (shared weights across fine/mid/coarse).
        - Mean-pool ViT tokens to obtain one feature vector per tubelet.
    """

    def __init__(self, vit3d_backbone: nn.Module):
        """
        Args:
            vit3d_backbone: A pretrained 3D ViT instance. Its weights
                are shared across all scales (fine/mid/coarse).
        """
        super().__init__()
        self.vit = vit3d_backbone

    def _encode_one_scale(
        self,
        tubelets: torch.Tensor,
        mask: torch.Tensor,
        channels: int = 1,
    ) -> torch.Tensor:
        """
        Encode tubelets for a single scale using the shared ViT3D.

        Args:
            tubelets: (B, N, D, H, W)
            mask:     (B, N) 1 for valid, 0 for padded
            channels: Number of channels expected by ViT (default: 1)
        Returns:
            features: (B, N, C_feat)
        """
        B, N, D, H, W = tubelets.shape

        # Merge batch and tubelet dimensions, add channel dim:
        # (B*N, C, H, W, D)
        video = tubelets.view(B * N, 1, D, H, W)
        video = video.repeat(1, channels, 1, 1, 1) if channels > 1 else video

        # Run through pretrained ViT3D
        tokens, _ = self.vit(video)  # (B*N, N_tokens, C_feat)

        # Simple mean pooling over tokens to get a single vector
        feat = tokens.mean(dim=1)  # (B*N, C_feat)

        # Restore (B, N, C_feat)
        feat = feat.view(B, N, -1)

        # Zero-out padded positions so downstream can safely aggregate
        if mask is not None:
            feat = feat * mask.unsqueeze(-1).to(feat.dtype)

        return feat

    def forward(
        self,
        tubelets_fine: torch.Tensor,
        tubelets_mid: torch.Tensor,
        tubelets_coarse: torch.Tensor,
        masks_fine: torch.Tensor,
        masks_mid: torch.Tensor,
        masks_coarse: torch.Tensor,
        channels: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            tubelets_fine:   (B, Nf, Df, Hf, Wf)
            tubelets_mid:    (B, Nm, Dm, Hm, Wm)
            tubelets_coarse: (B, Nc, Dc, Hc, Wc)
            masks_fine:      (B, Nf)
            masks_mid:       (B, Nm)
            masks_coarse:    (B, Nc)
            channels:        Number of channels expected by ViT3D
        Returns:
            F_fine:   (B, Nf, C_feat)
            F_mid:    (B, Nm, C_feat)
            F_coarse: (B, Nc, C_feat)
        """
        F_fine = self._encode_one_scale(tubelets_fine, masks_fine, channels=channels)
        F_mid = self._encode_one_scale(tubelets_mid, masks_mid, channels=channels)
        F_coarse = self._encode_one_scale(
            tubelets_coarse, masks_coarse, channels=channels
        )
        return F_fine, F_mid, F_coarse


__all__ = ["PretrainedViTVisionEncoder"]

