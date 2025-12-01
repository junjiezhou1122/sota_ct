#!/usr/bin/env python3
"""
CT Preprocessing Pipeline for Multi-Scale Tubelet Tokenization

Based on data quality findings:
- Handle extreme HU values (-8192, 7068)
- Region-specific HU windowing
- Proper Z-score normalization
- Multi-scale tubelet extraction

Author: Based on Data Quality Analysis 2025-11-29
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import torch
from dataclasses import dataclass
from enum import Enum


class AnatomyRegion(Enum):
    """Anatomical regions with specific HU windows
    
    Matches REGIONS from baseline:
    ['abdomen', 'bone', 'breast', 'esophagus', 'heart', 
     'lung', 'mediastinum', 'pleura', 'thyroid', 'trachea and bronchie']
    """
    ABDOMEN = "abdomen"
    BONE = "bone"
    BREAST = "breast"
    ESOPHAGUS = "esophagus"
    HEART = "heart"
    LUNG = "lung"
    MEDIASTINUM = "mediastinum"
    PLEURA = "pleura"
    THYROID = "thyroid"
    TRACHEA_BRONCHIE = "trachea and bronchie"
    GLOBAL = "global"  # For whole volume preprocessing


@dataclass
class HUWindow:
    """HU windowing parameters for different anatomical regions"""
    min_hu: float
    max_hu: float
    name: str
    
    @staticmethod
    def get_window(region: AnatomyRegion) -> 'HUWindow':
        """Get appropriate HU window for anatomical region"""
        windows = {
            AnatomyRegion.ABDOMEN: HUWindow(-160, 240, "abdomen"),
            AnatomyRegion.BONE: HUWindow(-650, 1350, "bone"),
            AnatomyRegion.BREAST: HUWindow(-160, 240, "breast"),
            AnatomyRegion.ESOPHAGUS: HUWindow(-160, 240, "esophagus"),
            AnatomyRegion.HEART: HUWindow(-160, 240, "heart"),
            AnatomyRegion.LUNG: HUWindow(-1350, 150, "lung"),
            AnatomyRegion.MEDIASTINUM: HUWindow(-160, 240, "mediastinum"),
            AnatomyRegion.PLEURA: HUWindow(-160, 240, "pleura"),  # Alternate: (-500, 500) if needed
            AnatomyRegion.THYROID: HUWindow(-160, 240, "thyroid"),
            AnatomyRegion.TRACHEA_BRONCHIE: HUWindow(-1350, 150, "trachea and bronchie"),
            AnatomyRegion.GLOBAL: HUWindow(-1024, 240, "global"), # Global windowing is important (capture the global info)
        }
        return windows[region]


@dataclass
class TubeletConfig:
    """Configuration for multi-scale tubelet extraction"""
    # Tubelet sizes (depth, height, width)
    fine_size: Tuple[int, int, int] = (16, 64, 64)      # Small receptive field
    mid_size: Tuple[int, int, int] = (32, 128, 128)     # Medium receptive field
    coarse_size: Tuple[int, int, int] = (64, 256, 256)  # Large receptive field (full context)
    
    # Stride (for overlapping tubelets)
    fine_stride: Tuple[int, int, int] = (8, 32, 32)     # 50% overlap
    mid_stride: Tuple[int, int, int] = (16, 64, 64)     # 50% overlap
    coarse_stride: Tuple[int, int, int] = (32, 128, 128) # 50% overlap
    
    # Target volume size (before tubelet extraction)
    target_shape: Tuple[int, int, int] = (128, 384, 384)  # (D, H, W)
    
    # Whether to use overlapping tubelets
    use_overlap: bool = True


class CTPreprocessor:
    """
    Comprehensive CT preprocessing pipeline
    
    Steps:
    1. Load NIfTI volume
    2. Handle extreme HU values (outliers)
    3. Apply region-specific HU windowing
    4. Z-score normalization
    5. Resize/resample to target shape
    6. Extract multi-scale tubelets
    """
    
    def __init__(
        self,
        tubelet_config: Optional[TubeletConfig] = None,
        region: AnatomyRegion = AnatomyRegion.GLOBAL,
        normalize_per_volume: bool = True
    ):
        self.tubelet_config = tubelet_config or TubeletConfig()
        self.region = region
        self.hu_window = HUWindow.get_window(region)
        self.normalize_per_volume = normalize_per_volume
        
    def load_volume(self, nifti_path: Path) -> Tuple[np.ndarray, Dict]:
        """Load NIfTI volume and metadata"""
        img = nib.load(str(nifti_path))
        volume = img.get_fdata()
        
        metadata = {
            'path': str(nifti_path),
            'original_shape': volume.shape,
            'spacing': img.header.get_zooms(),
            'affine': img.affine,
        }
        
        return volume, metadata
    
    def clip_hu_values(self, volume: np.ndarray) -> np.ndarray:
        """
        Clip HU values to reasonable range
        
        Handles extreme outliers found in inspection:
        - -8192 (likely padding/error values)
        - 7068+ (likely metal artifacts)
        """
        # Use region-specific window
        volume_clipped = np.clip(
            volume, 
            self.hu_window.min_hu, 
            self.hu_window.max_hu
        )
        
        return volume_clipped
    
    def normalize_volume(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Z-score normalization (per-volume or per-region)
        
        This is CORRECT normalization (not the linear scaling (x+400)/600)
        Results in zero-mean, unit-variance distribution
        """
        if mask is not None:
            # Normalize only within masked region
            masked_values = volume[mask > 0]
            if len(masked_values) == 0:
                return volume
            mean = masked_values.mean()
            std = masked_values.std()
        else:
            # Normalize entire volume
            mean = volume.mean()
            std = volume.std()
        
        # Avoid division by zero
        if std < 1e-8:
            std = 1.0
        
        volume_normalized = (volume - mean) / std
        
        return volume_normalized
    
    def resize_volume(
        self,
        volume: np.ndarray,
        target_shape: Optional[Tuple[int, int, int]] = None,
    ) -> np.ndarray:
        """Resize volume to target shape using pad/crop (preserves resolution)"""
        if target_shape is None:
            target_shape = self.tubelet_config.target_shape
        
        current_shape = volume.shape
        
        # Pad or crop to target size (preserves resolution)
        volume_resized = self._pad_or_crop(volume, target_shape)
        
        return volume_resized
    
    def _pad_or_crop(self, volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
        """Pad or center-crop volume to target shape"""
        current_shape = np.array(volume.shape)
        target_shape = np.array(target_shape)
        
        # Calculate padding/cropping for each dimension
        result = volume.copy()
        
        for dim in range(3):
            diff = target_shape[dim] - current_shape[dim]
            
            if diff > 0:
                # Need padding
                pad_before = diff // 2
                pad_after = diff - pad_before
                pad_width = [(0, 0)] * 3
                pad_width[dim] = (pad_before, pad_after)
                result = np.pad(result, pad_width, mode='constant', constant_values=-1024)
            elif diff < 0:
                # Need cropping
                crop_before = -diff // 2
                crop_after = current_shape[dim] - (-diff - crop_before)
                slices = [slice(None)] * 3
                slices[dim] = slice(crop_before, crop_after)
                result = result[tuple(slices)]
        
        return result
    
    def extract_tubelets(
        self, 
        volume: np.ndarray, 
        size: Tuple[int, int, int],
        stride: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Extract tubelets from volume
        
        Returns:
            tubelets: (N, D, H, W) where N is number of tubelets
        """
        if stride is None:
            stride = size  # Non-overlapping
        
        d, h, w = volume.shape
        td, th, tw = size
        sd, sh, sw = stride
        
        tubelets = []
        
        # Extract tubelets with sliding window
        for z in range(0, d - td + 1, sd):
            for y in range(0, h - th + 1, sh):
                for x in range(0, w - tw + 1, sw):
                    tubelet = volume[z:z+td, y:y+th, x:x+tw]
                    tubelets.append(tubelet)
        
        if len(tubelets) == 0:
            # Volume too small, return padded single tubelet
            return np.expand_dims(self._pad_or_crop(volume, size), 0)
        
        tubelets = np.stack(tubelets, axis=0)
        return tubelets
    
    def extract_multiscale_tubelets(self, volume: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract tubelets at three scales: fine, mid, coarse
        
        Returns:
            {
                'fine': (N1, 16, 64, 64),
                'mid': (N2, 32, 128, 128),
                'coarse': (N3, 64, 256, 256)
            }
        """
        config = self.tubelet_config
        
        tubelets = {}
        
        # Fine scale (small receptive field, many tubelets)
        tubelets['fine'] = self.extract_tubelets(
            volume, 
            config.fine_size,
            config.fine_stride if config.use_overlap else None
        )
        
        # Mid scale (medium receptive field)
        tubelets['mid'] = self.extract_tubelets(
            volume,
            config.mid_size,
            config.mid_stride if config.use_overlap else None
        )
        
        # Coarse scale (large receptive field, few tubelets)
        # For full context, might just be the whole volume resized
        tubelets['coarse'] = self.extract_tubelets(
            volume,
            config.coarse_size,
            config.coarse_stride if config.use_overlap else None
        )
        
        return tubelets
    
    def preprocess(
        self, 
        nifti_path: Path,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Full preprocessing pipeline
        
        Returns:
            {
                'volume': preprocessed volume (D, H, W),
                'tubelets_fine': (N1, 16, 64, 64),
                'tubelets_mid': (N2, 32, 128, 128),
                'tubelets_coarse': (N3, 64, 256, 256),
                'metadata': {...}
            }
        """
        # 1. Load
        volume, metadata = self.load_volume(nifti_path)
        
        # 2. Clip extreme HU values
        volume = self.clip_hu_values(volume)
        
        # 3. Normalize
        volume = self.normalize_volume(volume, mask)
        
        # 4. Resize to target shape
        volume = self.resize_volume(volume)
        
        # 5. Extract multi-scale tubelets
        tubelets = self.extract_multiscale_tubelets(volume)
        
        result = {
            'volume': volume,
            'tubelets_fine': tubelets['fine'],
            'tubelets_mid': tubelets['mid'],
            'tubelets_coarse': tubelets['coarse'],
            'metadata': metadata
        }
        
        return result


class RegionSpecificPreprocessor:
    """
    Preprocessor that handles multiple anatomical regions with different HU windows
    """
    
    def __init__(self, tubelet_config: Optional[TubeletConfig] = None):
        self.tubelet_config = tubelet_config or TubeletConfig()
        self.preprocessors = {
            region: CTPreprocessor(tubelet_config, region)
            for region in AnatomyRegion
        }
    
    def preprocess_with_masks(
        self,
        nifti_path: Path,
        region_masks: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Preprocess volume with region-specific windowing
        
        Args:
            nifti_path: Path to CT NIfTI file
            region_masks: Dict of region_name -> binary mask
        
        Returns:
            {
                'global': {...},  # Global preprocessing
                'lung': {...},    # Lung-specific preprocessing
                'heart': {...},   # Heart-specific preprocessing
                ...
            }
        """
        results = {}
        
        # Global preprocessing
        global_preprocessor = self.preprocessors[AnatomyRegion.GLOBAL]
        results['global'] = global_preprocessor.preprocess(nifti_path)
        
        # Load original volume
        volume, _ = global_preprocessor.load_volume(nifti_path)
        
        # Region-specific preprocessing
        for region_name, mask in region_masks.items():
            try:
                # Map region name to enum
                region_enum = AnatomyRegion[region_name.upper()]
                preprocessor = self.preprocessors[region_enum]
                
                # Apply mask to volume
                masked_volume = volume.copy()
                masked_volume[mask == 0] = -1024  # Set non-region to air HU
                
                # Preprocess with region-specific window
                results[region_name] = preprocessor.preprocess(nifti_path, mask=mask)
                
            except KeyError:
                print(f"Warning: Unknown region {region_name}, skipping")
                continue
        
        return results

