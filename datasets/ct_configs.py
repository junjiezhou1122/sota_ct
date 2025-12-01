#!/usr/bin/env python3
"""
Configuration objects for CT preprocessing and tubelet extraction.

This file centralizes all tubelet-related hyperparameters so that:
- the preprocessing and dataset logic stays model-agnostic;
- different backbones (ViT variants, etc.) can be supported by
  simply defining new config helpers.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TubeletConfig:
    """Configuration for multi-scale tubelet extraction."""

    # Tubelet sizes (depth, height, width)
    fine_size: Tuple[int, int, int] = (16, 64, 64)      # Small receptive field
    mid_size: Tuple[int, int, int] = (32, 128, 128)     # Medium receptive field
    coarse_size: Tuple[int, int, int] = (64, 256, 256)  # Large receptive field (full context)

    # Stride (for overlapping tubelets)
    fine_stride: Tuple[int, int, int] = (8, 32, 32)      # 50% overlap
    mid_stride: Tuple[int, int, int] = (16, 64, 64)      # 50% overlap
    coarse_stride: Tuple[int, int, int] = (32, 128, 128) # 50% overlap

    # Target volume size (before tubelet extraction)
    target_shape: Tuple[int, int, int] = (128, 384, 384)  # (D, H, W)

    # Whether to use overlapping tubelets
    use_overlap: bool = True


def default_config_for_paper() -> TubeletConfig:
    """
    Tubelet configuration used in our paper experiments.

    If you change the backbone (e.g., deeper 3D ViT, different
    patch sizes), define a new helper here instead of touching
    the preprocessing / dataset code.
    """
    return TubeletConfig(
        fine_size=(16, 64, 64),
        mid_size=(32, 128, 128),
        coarse_size=(64, 256, 256),
        fine_stride=(8, 32, 32),
        mid_stride=(16, 64, 64),
        coarse_stride=(32, 128, 128),
        target_shape=(128, 384, 384),
        use_overlap=True,
    )

