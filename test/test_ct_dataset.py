#!/usr/bin/env python3
"""
Simple sanity checks for CTReportDataset and dataloaders.

Run:
    python test/test_ct_dataset.py
"""

from pathlib import Path

import torch

from datasets.ct_dataset import CTReportDataset, create_dataloaders


def test_ct_report_dataset():
    """Basic smoke test for CTReportDataset (train split)."""
    # Adjust these paths to your local setup
    train_data_dir = Path("/mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed")
    train_csv_file = Path(
        "/mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv"
    )
    mask_dir = Path("/mnt2/ct/RadGenome-ChestCT/dataset/train_masks")

    dataset = CTReportDataset(
        data_dir=train_data_dir,
        csv_file=train_csv_file,
        mask_dir=mask_dir,
        use_regions=True,
        augment=True,
        max_samples=10,  # small subset for quick check
    )

    sample = dataset[0]

    print("Sample loaded successfully!")
    print(f"Fine tubelets: {sample['tubelets_fine'].shape}")
    print(f"Mid tubelets: {sample['tubelets_mid'].shape}")
    print(f"Coarse tubelets: {sample['tubelets_coarse'].shape}")
    print(f"Volume: {sample['volume'].shape}")
    print(f"Regions: {list(sample['regions'].keys())}")


def test_dataloaders():
    """Basic smoke test for train/val dataloaders using official split."""
    train_data_dir = Path("/mnt2/ct/RadGenome-ChestCT/dataset/train_preprocessed")
    train_csv_file = Path(
        "/mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/train_region_report.csv"
    )
    val_data_dir = Path("/mnt2/ct/RadGenome-ChestCT/dataset/valid_preprocessed")
    val_csv_file = Path(
        "/mnt2/ct/RadGenome-ChestCT/dataset/radgenome_files/validation_region_report.csv"
    )
    mask_dir = Path("/mnt2/ct/RadGenome-ChestCT/dataset/train_masks")

    train_loader, val_loader = create_dataloaders(
        train_data_dir=train_data_dir,
        train_csv_file=train_csv_file,
        val_data_dir=val_data_dir,
        val_csv_file=val_csv_file,
        mask_dir=mask_dir,
        batch_size=2,
        num_workers=0,
    )

    batch = next(iter(train_loader))
    print("\nTrain batch loaded successfully!")
    print(f"Batch volumes: {batch['volumes'].shape}")
    print(f"Batch fine tubelets: {batch['tubelets_fine'].shape}")
    print(f"Batch masks (fine): {batch['masks_fine'].shape}")


if __name__ == "__main__":
    test_ct_report_dataset()
    test_dataloaders()

