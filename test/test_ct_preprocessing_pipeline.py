#!/usr/bin/env python3
"""
Simple example usage for CT preprocessing pipeline.

Run:
    python test/test_ct_preprocessing_pipeline.py
"""

from pathlib import Path

import torch

from datasets.ct_preprocessing_pipeline import (
    CTPreprocessor,
    AnatomyRegion,
)
from datasets.ct_configs import default_config_for_paper


def example_preprocess_single_volume():
    """Run preprocessing on a single volume and print shapes."""
    config = default_config_for_paper()

    preprocessor = CTPreprocessor(
        tubelet_config=config,
        region=AnatomyRegion.GLOBAL,
    )

    # TODO: adjust this path to point to an existing NIfTI file
    nifti_path = Path("/mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed/train_1/train_1a/train_1_a_1.nii.gz")

    result = preprocessor.preprocess(nifti_path)

    print("Preprocessing completed.")
    print(f"Preprocessed volume shape: {result['volume'].shape}")
    print(f"Fine tubelets: {result['tubelets_fine'].shape}")
    print(f"Mid tubelets: {result['tubelets_mid'].shape}")
    print(f"Coarse tubelets: {result['tubelets_coarse'].shape}")

    # Convert to tensors (example)
    volume_tensor = torch.from_numpy(result['volume']).float()
    fine_tensor = torch.from_numpy(result['tubelets_fine']).float()
    print(f"Volume tensor dtype: {volume_tensor.dtype}, device: {volume_tensor.device}")
    print(f"Fine tubelets tensor dtype: {fine_tensor.dtype}, device: {fine_tensor.device}")


if __name__ == "__main__":
    example_preprocess_single_volume()
