#!/usr/bin/env python3
"""
PyTorch Dataset for CT Report Generation with Multi-Scale Tubelets

Integrates with the preprocessing pipeline and handles:
- Multi-scale tubelet tokenization
- Region-specific processing
- Data augmentation
- Efficient caching
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from ct_preprocessing_pipeline import (
    CTPreprocessor,
    RegionSpecificPreprocessor,
    AnatomyRegion,
)
from ct_configs import TubeletConfig, default_config_for_paper


class CTReportDataset(Dataset):
    """
    Dataset for CT report generation with multi-scale tubelets
    
    Features:
    - Multi-scale tubelet extraction (fine/mid/coarse)
    - Region-specific HU windowing
    - Proper Z-score normalization
    - Optional data augmentation
    - Memory-efficient loading
    """
    
    def __init__(
        self,
        data_dir: Path,
        csv_file: Path,
        mask_dir: Optional[Path] = None,
        tubelet_config: Optional[TubeletConfig] = None,
        use_regions: bool = True,
        augment: bool = False,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            data_dir: Directory containing NIfTI files
            csv_file: CSV with columns: Volumename, Anatomy, Sentence
            mask_dir: Directory containing region masks
            tubelet_config: Configuration for tubelet extraction
            use_regions: Whether to use region-specific preprocessing
            augment: Whether to apply data augmentation
            max_samples: Limit dataset size (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        # Use the paper default unless an explicit config is provided
        self.tubelet_config = tubelet_config or default_config_for_paper()
        self.use_regions = use_regions
        self.augment = augment
        
        # Initialize preprocessors
        if use_regions:
            self.preprocessor = RegionSpecificPreprocessor(self.tubelet_config)
        else:
            self.preprocessor = CTPreprocessor(
                self.tubelet_config, 
                region=AnatomyRegion.GLOBAL
            )
        
        # Load dataset metadata
        self.samples = self._load_samples(csv_file, max_samples)
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _load_samples(self, csv_file: Path, max_samples: Optional[int]) -> List[Dict]:
        """Load sample list from CSV"""
        df = pd.read_csv(csv_file)
        
        # Group by volume
        df_grouped = df.groupby('Volumename')
        
        samples = []
        for volume_name, group in df_grouped:
            # Find NIfTI file
            nifti_file = self.data_dir / volume_name
            if not nifti_file.exists():
                continue
            
            # Collect region reports
            regions = {}
            for _, row in group.iterrows():
                anatomy = row.get('Anatomy', 'whole')
                if pd.isna(anatomy):
                    anatomy = 'whole'
                sentence = row.get('Sentence', '')
                regions[anatomy] = sentence
            
            sample = {
                'nifti_path': nifti_file,
                'volume_name': volume_name,
                'regions': regions
            }
            
            samples.append(sample)
            
            if max_samples and len(samples) >= max_samples:
                break
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_region_masks(self, volume_name: str) -> Dict[str, np.ndarray]:
        """Load region masks for a volume"""
        if self.mask_dir is None:
            return {}
        
        masks = {}
        mask_folder = self.mask_dir / f"seg_{volume_name.split('.')[0]}"
        
        if not mask_folder.exists():
            return {}
        
        # Load masks for each region
        for mask_file in mask_folder.glob("*.nii.gz"):
            region_name = mask_file.stem  # e.g., 'lung', 'heart'
            try:
                import nibabel as nib
                mask_data = nib.load(str(mask_file)).get_fdata()
                masks[region_name] = mask_data
            except Exception as e:
                print(f"Warning: Failed to load mask {mask_file}: {e}")
                continue
        
        return masks
    
    def _augment_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation
        
        Augmentations:
        - Random flip (left-right)
        - Random rotation (small angle)
        - Random intensity shift
        - Random noise
        """
        if not self.augment:
            return volume
        
        # Random flip (50% chance)
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=2).copy()  # Flip width
        
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=1).copy()  # Flip height
        
        # Random rotation (small angle, ±10 degrees)
        if np.random.rand() > 0.7:
            from scipy.ndimage import rotate
            angle = np.random.uniform(-10, 10)
            # Rotate in axial plane
            volume = rotate(volume, angle, axes=(1, 2), reshape=False, order=1)
        
        # Random intensity shift (±10% of std)
        if np.random.rand() > 0.5:
            shift = np.random.normal(0, 0.1)
            volume = volume + shift
        
        # Random Gaussian noise
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.05, volume.shape)
            volume = volume + noise
        
        return volume
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a preprocessed sample
        
        Returns:
            {
                'tubelets_fine': (N1, 16, 64, 64),
                'tubelets_mid': (N2, 32, 128, 128),
                'tubelets_coarse': (N3, 64, 256, 256),
                'volume': (128, 384, 384),  # Optional, for visualization
                'regions': dict of region reports,
                'volume_name': str
            }
        """
        sample = self.samples[idx]
        nifti_path = sample['nifti_path']
        
        # Load region masks if using region-specific preprocessing
        if self.use_regions and isinstance(self.preprocessor, RegionSpecificPreprocessor):
            region_masks = self._load_region_masks(sample['volume_name'])
            result = self.preprocessor.preprocess_with_masks(nifti_path, region_masks)
            
            # Use global preprocessing result
            preprocessed = result['global']
        else:
            # Simple global preprocessing
            preprocessed = self.preprocessor.preprocess(nifti_path)
        
        # Apply augmentation
        volume = preprocessed['volume']
        if self.augment:
            volume = self._augment_volume(volume)
            # Re-extract tubelets from augmented volume
            tubelets = self.preprocessor.extract_multiscale_tubelets(volume)
            preprocessed['tubelets_fine'] = tubelets['fine']
            preprocessed['tubelets_mid'] = tubelets['mid']
            preprocessed['tubelets_coarse'] = tubelets['coarse']
        
        # Convert to tensors
        output = {
            'tubelets_fine': torch.from_numpy(preprocessed['tubelets_fine']).float(),
            'tubelets_mid': torch.from_numpy(preprocessed['tubelets_mid']).float(),
            'tubelets_coarse': torch.from_numpy(preprocessed['tubelets_coarse']).float(),
            'volume': torch.from_numpy(volume).float(),
            'regions': sample['regions'],
            'volume_name': sample['volume_name']
        }
        
        return output


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable number of tubelets
    
    Since different volumes may produce different numbers of tubelets,
    we need to handle this carefully.
    """
    # Stack volumes (all same size after preprocessing)
    volumes = torch.stack([item['volume'] for item in batch])
    
    # For tubelets, we have two options:
    # Option 1: Pad to max length in batch

    def pad_tubelets(tubelet_list, pad_value=-1024):
        """Pad tubelets to same length"""
        max_len = max(t.shape[0] for t in tubelet_list)
        padded = []
        masks = []
        
        for tubelets in tubelet_list:
            n_tubelets = tubelets.shape[0]
            if n_tubelets < max_len:
                # Pad
                pad_shape = (max_len - n_tubelets,) + tubelets.shape[1:]
                padding = torch.full(pad_shape, pad_value, dtype=tubelets.dtype)
                tubelets_padded = torch.cat([tubelets, padding], dim=0)
                
                # Create mask (1 for real, 0 for padded)
                mask = torch.cat([
                    torch.ones(n_tubelets),
                    torch.zeros(max_len - n_tubelets)
                ])
            else:
                tubelets_padded = tubelets
                mask = torch.ones(n_tubelets)
            
            padded.append(tubelets_padded)
            masks.append(mask)
        
        return torch.stack(padded), torch.stack(masks)
    
    tubelets_fine, masks_fine = pad_tubelets([item['tubelets_fine'] for item in batch])
    tubelets_mid, masks_mid = pad_tubelets([item['tubelets_mid'] for item in batch])
    tubelets_coarse, masks_coarse = pad_tubelets([item['tubelets_coarse'] for item in batch])
    
    # Collect regions and names
    regions = [item['regions'] for item in batch]
    volume_names = [item['volume_name'] for item in batch]
    
    return {
        'volumes': volumes,
        'tubelets_fine': tubelets_fine,
        'tubelets_mid': tubelets_mid,
        'tubelets_coarse': tubelets_coarse,
        'masks_fine': masks_fine,
        'masks_mid': masks_mid,
        'masks_coarse': masks_coarse,
        'regions': regions,
        'volume_names': volume_names
    }


def create_dataloaders(
    train_data_dir: Path,
    train_csv_file: Path,
    val_data_dir: Path,
    val_csv_file: Path,
    mask_dir: Optional[Path] = None,
    batch_size: int = 4,
    num_workers: int = 4,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders using the official
    RadGenome train/validation split.
    """
    # Training dataset (with augmentation)
    train_dataset = CTReportDataset(
        data_dir=train_data_dir,
        csv_file=train_csv_file,
        mask_dir=mask_dir,
        augment=True
    )
    
    # Validation dataset (no augmentation)
    val_dataset = CTReportDataset(
        data_dir=val_data_dir,
        csv_file=val_csv_file,
        mask_dir=mask_dir,
        augment=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
