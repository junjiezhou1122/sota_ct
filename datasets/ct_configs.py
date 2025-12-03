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
    # Sizes tuned for current dataset stats (D/H≈370, W≈100, spacing=1mm)
    # Use narrower W to avoid padding huge empty regions.
    fine_size: Tuple[int, int, int] = (16, 64, 64)       # Local detail
    mid_size: Tuple[int, int, int] = (32, 96, 96)        # Regional context
    coarse_size: Tuple[int, int, int] = (48, 160, 160)   # Global-ish context

    # Stride (overlap ~50%)
    fine_stride: Tuple[int, int, int] = (8, 32, 32)
    mid_stride: Tuple[int, int, int] = (16, 48, 48)
    coarse_stride: Tuple[int, int, int] = (24, 80, 80)

    # Target volume size (before tubelet extraction), multiple of 16 for clean grids
    target_shape: Tuple[int, int, int] = (192, 384, 160)  # (D, H, W)

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
        mid_size=(32, 96, 96),
        coarse_size=(48, 160, 160),
        fine_stride=(8, 32, 32),
        mid_stride=(16, 48, 48),
        coarse_stride=(24, 80, 80),
        target_shape=(192, 384, 160),
        use_overlap=True,  # keep overlap to densify coverage
    )
