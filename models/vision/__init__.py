#!/usr/bin/env python3
"""
Vision encoders for multi-scale CT tubelets.

This package provides:
    - TubeletEmbedding3D: low-level tubelet -> vector embedding
    - SharedViTEncoder: shared Transformer encoder over tubelet tokens
    - VisionEncoder: end-to-end shared vision encoder (CNN + Transformer)
    - PretrainedViTVisionEncoder: wrapper around a pretrained 3D ViT
"""

from .tubelet_embedding import TubeletEmbedding3D
from .shared_transformer import SharedViTEncoder
from .shared_vision_encoder import VisionEncoder
from .pretrained_vit_encoder import PretrainedViTVisionEncoder

__all__ = [
    "TubeletEmbedding3D",
    "SharedViTEncoder",
    "VisionEncoder",
    "PretrainedViTVisionEncoder",
]

