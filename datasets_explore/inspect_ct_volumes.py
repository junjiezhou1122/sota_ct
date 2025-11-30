#!/usr/bin/env python3
"""
Comprehensive CT Volume Inspection Script

This script performs thorough analysis of CT scan data to understand:
- Volume dimensions and spacing
- HU value distributions
- Data quality issues
- Preprocessing requirements

Usage:
    python inspect_ct_volumes.py --data_dir /path/to/ct/data --format dicom
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    print("Warning: matplotlib/seaborn not available. Skipping plots.")
    HAS_PLOTTING = False

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    print("Warning: SimpleITK not available. Limited functionality.")
    HAS_SITK = False

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    print("Warning: pydicom not available. Cannot read DICOM files.")
    HAS_PYDICOM = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    print("Warning: nibabel not available. Cannot read NIfTI files.")
    HAS_NIBABEL = False


class CTVolumeInspector:
    """Comprehensive CT volume data inspector"""
    
    def __init__(self, data_dir: str, output_dir: str = "inspection_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.volume_stats = []
        self.hu_stats = []
        self.quality_issues = []
        self.metadata_list = []
        
    def find_ct_files(self, format_type: str = "auto") -> List[Path]:
        """Find all CT files in the directory"""
        print(f"\n🔍 Searching for CT files in {self.data_dir}...")
        
        if format_type == "dicom" or format_type == "auto":
            # Look for DICOM directories (each patient/scan is a folder)
            dicom_dirs = []
            for item in self.data_dir.rglob("*"):
                if item.is_dir():
                    # Check if this directory contains DICOM files
                    dcm_files = list(item.glob("*.dcm")) + list(item.glob("*.DCM"))
                    if dcm_files:
                        dicom_dirs.append(item)
            
            if dicom_dirs:
                print(f"   Found {len(dicom_dirs)} DICOM series")
                return dicom_dirs
        
        if format_type == "nifti" or format_type == "auto":
            # Only accept proper NIfTI extensions to avoid matching arbitrary gzip files
            nifti_files = list(self.data_dir.rglob("*.nii.gz")) + list(self.data_dir.rglob("*.nii"))
            if nifti_files:
                print(f"   Found {len(nifti_files)} NIfTI files")
                return nifti_files
        
        if format_type == "npy" or format_type == "auto":
            npy_files = list(self.data_dir.rglob("*.npy")) + list(self.data_dir.rglob("*.npz"))
            if npy_files:
                print(f"   Found {len(npy_files)} NumPy files")
                return npy_files
        
        print("   ⚠️  No CT files found!")
        return []
    
    def load_dicom_volume(self, dicom_dir: Path) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Load a DICOM series as a 3D volume"""
        if not HAS_SITK or not HAS_PYDICOM:
            return None, None
        
        try:
            # Use SimpleITK to read DICOM series
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
            
            if not dicom_names:
                return None, None
            
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            
            # Get volume array
            volume = sitk.GetArrayFromImage(image)  # Shape: (D, H, W)
            
            # Get metadata
            spacing = image.GetSpacing()  # (x, y, z)
            origin = image.GetOrigin()
            direction = image.GetDirection()
            
            # Read DICOM metadata from first slice
            dcm = pydicom.dcmread(dicom_names[0], force=True)
            
            metadata = {
                'path': str(dicom_dir),
                'shape': volume.shape,
                'spacing': spacing,  # (x_spacing, y_spacing, z_spacing) in mm
                'origin': origin,
                'modality': getattr(dcm, 'Modality', 'Unknown'),
                'manufacturer': getattr(dcm, 'Manufacturer', 'Unknown'),
                'slice_thickness': getattr(dcm, 'SliceThickness', spacing[2]),
                'kvp': getattr(dcm, 'KVP', None),
                'exposure': getattr(dcm, 'Exposure', None),
                'kernel': getattr(dcm, 'ConvolutionKernel', 'Unknown'),
                'num_slices': len(dicom_names)
            }
            
            return volume, metadata
            
        except Exception as e:
            print(f"   ⚠️  Error loading {dicom_dir}: {e}")
            self.quality_issues.append({
                'path': str(dicom_dir),
                'issue': 'load_error',
                'details': str(e)
            })
            return None, None
    
    def load_nifti_volume(self, nifti_path: Path) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Load a NIfTI file as a 3D volume"""
        if not HAS_NIBABEL:
            return None, None
        
        try:
            img = nib.load(str(nifti_path))
            volume = img.get_fdata()
            
            # Get spacing from affine matrix
            spacing = img.header.get_zooms()
            
            metadata = {
                'path': str(nifti_path),
                'shape': volume.shape,
                'spacing': spacing,  # (x, y, z) in mm
                'affine': img.affine.tolist(),
                'dtype': str(volume.dtype)
            }
            
            return volume, metadata
            
        except Exception as e:
            print(f"   ⚠️  Error loading {nifti_path}: {e}")
            self.quality_issues.append({
                'path': str(nifti_path),
                'issue': 'load_error',
                'details': str(e)
            })
            return None, None
    
    def load_numpy_volume(self, npy_path: Path) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Load a NumPy file"""
        try:
            if npy_path.suffix == '.npz':
                data = np.load(str(npy_path))
                # Try common keys
                for key in ['volume', 'data', 'image', 'ct']:
                    if key in data:
                        volume = data[key]
                        break
                else:
                    volume = data[list(data.keys())[0]]
            else:
                volume = np.load(str(npy_path))
            
            metadata = {
                'path': str(npy_path),
                'shape': volume.shape,
                'dtype': str(volume.dtype)
            }
            
            return volume, metadata
            
        except Exception as e:
            print(f"   ⚠️  Error loading {npy_path}: {e}")
            self.quality_issues.append({
                'path': str(npy_path),
                'issue': 'load_error',
                'details': str(e)
            })
            return None, None
    
    def analyze_volume(self, volume: np.ndarray, metadata: Dict) -> Dict:
        """Analyze a single CT volume"""
        stats = {
            'path': metadata['path'],
            'shape_d': volume.shape[0],
            'shape_h': volume.shape[1] if len(volume.shape) > 1 else 1,
            'shape_w': volume.shape[2] if len(volume.shape) > 2 else 1,
        }
        
        # Spacing analysis
        if 'spacing' in metadata:
            spacing = metadata['spacing']
            stats['spacing_x'] = spacing[0] if len(spacing) > 0 else None
            stats['spacing_y'] = spacing[1] if len(spacing) > 1 else None
            stats['spacing_z'] = spacing[2] if len(spacing) > 2 else None
            
            # Anisotropy ratio
            if len(spacing) == 3:
                stats['anisotropy_xy'] = spacing[0] / spacing[1] if spacing[1] > 0 else 1.0
                stats['anisotropy_z_to_xy'] = spacing[2] / ((spacing[0] + spacing[1]) / 2)
        
        # HU value statistics (sample to avoid memory issues)
        sample_size = min(volume.size, 10_000_000)
        if volume.size > sample_size:
            # Use strided sampling to avoid allocating a huge index array
            step = max(1, volume.size // sample_size)
            sample = volume.ravel()[::step][:sample_size]
        else:
            sample = volume.ravel()
        
        stats['hu_min'] = float(np.min(sample))
        stats['hu_max'] = float(np.max(sample))
        stats['hu_mean'] = float(np.mean(sample))
        stats['hu_std'] = float(np.std(sample))
        stats['hu_p01'] = float(np.percentile(sample, 1))
        stats['hu_p05'] = float(np.percentile(sample, 5))
        stats['hu_p25'] = float(np.percentile(sample, 25))
        stats['hu_median'] = float(np.median(sample))
        stats['hu_p75'] = float(np.percentile(sample, 75))
        stats['hu_p95'] = float(np.percentile(sample, 95))
        stats['hu_p99'] = float(np.percentile(sample, 99))
        
        # Quality checks
        issues = []
        
        # Check for extreme values
        if stats['hu_min'] < -2000:
            issues.append(f"Extreme low HU: {stats['hu_min']}")
        if stats['hu_max'] > 5000:
            issues.append(f"Extreme high HU: {stats['hu_max']}")
        
        # Check for too few slices
        if stats['shape_d'] < 20:
            issues.append(f"Too few slices: {stats['shape_d']}")
        
        # Check for unusual dimensions
        if stats['shape_h'] < 256 or stats['shape_w'] < 256:
            issues.append(f"Small image size: {stats['shape_h']}x{stats['shape_w']}")
        
        # Check for extreme anisotropy
        if 'anisotropy_z_to_xy' in stats and stats['anisotropy_z_to_xy'] > 5:
            issues.append(f"High anisotropy: {stats['anisotropy_z_to_xy']:.2f}")
        
        if issues:
            stats['quality_issues'] = '; '.join(issues)
            self.quality_issues.append({
                'path': metadata['path'],
                'issue': 'quality_check',
                'details': '; '.join(issues)
            })
        
        # Add metadata
        for key in ['modality', 'manufacturer', 'kernel']:
            if key in metadata:
                stats[key] = metadata[key]
        
        return stats
    
    def inspect_dataset(self, format_type: str = "auto", max_samples: Optional[int] = None, batch_save: int = 100):
        """Inspect all CT volumes in the dataset"""
        print("\n" + "="*60)
        print("CT VOLUME INSPECTION")
        print("="*60)
        
        # Find files
        ct_files = self.find_ct_files(format_type)
        
        if not ct_files:
            print("\n❌ No CT files found. Please check the data directory.")
            return
        
        # Limit samples if specified
        if max_samples:
            ct_files = ct_files[:max_samples]
            print(f"\n📊 Inspecting {len(ct_files)} samples (limited to {max_samples})")
        else:
            print(f"\n📊 Inspecting {len(ct_files)} volumes...")
        
        # Track progress
        start_time = time.time()
        error_count = 0
        
        # Process each file with progress bar
        for i, ct_path in enumerate(tqdm(ct_files, desc="Processing volumes", unit="vol"), 1):
            try:
                # Load volume
                volume, metadata = None, None
                
                if ct_path.is_dir():  # DICOM
                    volume, metadata = self.load_dicom_volume(ct_path)
                elif ct_path.name.endswith('.nii.gz') or ct_path.suffix == '.nii':  # NIfTI
                    volume, metadata = self.load_nifti_volume(ct_path)
                elif ct_path.suffix in ['.npy', '.npz']:  # NumPy
                    volume, metadata = self.load_numpy_volume(ct_path)
                
                if volume is None or metadata is None:
                    error_count += 1
                    continue
                
                # Analyze
                stats = self.analyze_volume(volume, metadata)
                self.volume_stats.append(stats)
                self.metadata_list.append(metadata)
                
                # Batch save intermediate results (in case of interruption)
                if i % batch_save == 0:
                    self._save_intermediate_results(i)
                    
            except Exception as e:
                error_count += 1
                tqdm.write(f"⚠️  Error processing {ct_path.name}: {e}")
                self.quality_issues.append({
                    'path': str(ct_path),
                    'issue': 'processing_error',
                    'details': str(e)
                })
                continue
        
        elapsed_time = time.time() - start_time
        processed = max(1, len(self.volume_stats))
        print(f"\n✅ Completed inspection of {len(self.volume_stats)} volumes")
        print(f"   Time elapsed: {elapsed_time/60:.1f} minutes")
        print(f"   Average time per volume: {elapsed_time/processed:.2f} seconds")
        if error_count > 0:
            print(f"   ⚠️  Errors encountered: {error_count} volumes")
    
    def _save_intermediate_results(self, current_index: int):
        """Save intermediate results during processing"""
        if not self.volume_stats:
            return
        
        temp_df = pd.DataFrame(self.volume_stats)
        temp_path = self.output_dir / f"volume_statistics_temp_{current_index}.csv"
        temp_df.to_csv(temp_path, index=False)
        
        # Save quality issues so far
        if self.quality_issues:
            temp_issues_path = self.output_dir / f"quality_issues_temp_{current_index}.json"
            with open(temp_issues_path, 'w') as f:
                json.dump(self.quality_issues, f, indent=2)
    
    def generate_summary(self):
        """Generate summary statistics"""
        if not self.volume_stats:
            print("\n❌ No volume statistics available")
            return
        
        df = pd.DataFrame(self.volume_stats)
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Basic counts
        print(f"\n📊 Dataset Overview:")
        print(f"   Total volumes: {len(df)}")
        print(f"   Quality issues: {len(self.quality_issues)}")
        
        # Shape statistics
        print(f"\n📐 Volume Dimensions:")
        print(f"   Slices (D):  min={df['shape_d'].min()}, max={df['shape_d'].max()}, median={df['shape_d'].median():.0f}")
        print(f"   Height (H):  min={df['shape_h'].min()}, max={df['shape_h'].max()}, median={df['shape_h'].median():.0f}")
        print(f"   Width (W):   min={df['shape_w'].min()}, max={df['shape_w'].max()}, median={df['shape_w'].median():.0f}")
        
        # Spacing statistics
        if 'spacing_x' in df.columns:
            print(f"\n📏 Voxel Spacing (mm):")
            print(f"   X (in-plane):    min={df['spacing_x'].min():.3f}, max={df['spacing_x'].max():.3f}, median={df['spacing_x'].median():.3f}")
            print(f"   Y (in-plane):    min={df['spacing_y'].min():.3f}, max={df['spacing_y'].max():.3f}, median={df['spacing_y'].median():.3f}")
            print(f"   Z (thickness):   min={df['spacing_z'].min():.3f}, max={df['spacing_z'].max():.3f}, median={df['spacing_z'].median():.3f}")
        
        if 'anisotropy_z_to_xy' in df.columns:
            print(f"\n📊 Anisotropy Ratio (Z/XY):")
            print(f"   min={df['anisotropy_z_to_xy'].min():.2f}, max={df['anisotropy_z_to_xy'].max():.2f}, median={df['anisotropy_z_to_xy'].median():.2f}")
        
        # HU value statistics
        print(f"\n🔬 HU Value Distribution:")
        print(f"   Global min:  {df['hu_min'].min():.1f}")
        print(f"   Global max:  {df['hu_max'].max():.1f}")
        print(f"   Avg mean:    {df['hu_mean'].mean():.1f} ± {df['hu_mean'].std():.1f}")
        print(f"   1st percentile:  {df['hu_p01'].median():.1f}")
        print(f"   5th percentile:  {df['hu_p05'].median():.1f}")
        print(f"   95th percentile: {df['hu_p95'].median():.1f}")
        print(f"   99th percentile: {df['hu_p99'].median():.1f}")
        
        # Metadata
        if 'manufacturer' in df.columns:
            print(f"\n🏥 Scanner Info:")
            print("   Manufacturers:")
            for mfr, count in df['manufacturer'].value_counts().items():
                print(f"      {mfr}: {count}")
        
        if 'kernel' in df.columns:
            print("   Reconstruction Kernels:")
            for kernel, count in df['kernel'].value_counts().head(5).items():
                print(f"      {kernel}: {count}")
        
        # Quality issues summary
        if self.quality_issues:
            print(f"\n⚠️  Quality Issues Summary:")
            issue_types = defaultdict(int)
            for issue in self.quality_issues:
                issue_types[issue['issue']] += 1
            for issue_type, count in issue_types.items():
                print(f"   {issue_type}: {count}")
        
        # Recommendations
        print(f"\n💡 Preprocessing Recommendations:")
        
        # HU clipping
        p01 = df['hu_p01'].median()
        p99 = df['hu_p99'].median()
        print(f"   1. HU Clipping: [{p01:.0f}, {p99:.0f}] (based on 1st-99th percentiles)")
        print(f"      Alternative lung window: [-1000, 400]")
        print(f"      Alternative soft tissue: [-160, 240]")
        
        # Resampling
        if 'spacing_x' in df.columns:
            median_xy = (df['spacing_x'].median() + df['spacing_y'].median()) / 2
            median_z = df['spacing_z'].median()
            print(f"   2. Target Spacing: {median_xy:.2f} × {median_xy:.2f} × {median_z:.2f} mm")
            print(f"      (based on dataset median)")
        
        # Slice handling
        median_slices = df['shape_d'].median()
        max_slices = df['shape_d'].max()
        print(f"   3. Slice Count: median={median_slices:.0f}, max={max_slices}")
        print(f"      Suggest target: 128 or 160 slices")
        print(f"      Volumes > target: use sliding window")
        print(f"      Volumes < target: use padding or interpolation")
        
        # Save summary
        summary_path = self.output_dir / "dataset_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("CT DATASET INSPECTION SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total volumes: {len(df)}\n")
            f.write(f"Quality issues: {len(self.quality_issues)}\n\n")
            f.write(df.describe().to_string())
        
        print(f"\n💾 Summary saved to: {summary_path}")
        
        return df
    
    def generate_plots(self, df: pd.DataFrame):
        """Generate visualization plots"""
        if not HAS_PLOTTING:
            print("\n⚠️  Plotting libraries not available. Skipping plots.")
            return
        
        print("\n📈 Generating plots...")
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle("CT Dataset Inspection", fontsize=16, fontweight='bold')
        
        # Shape distributions
        axes[0, 0].hist(df['shape_d'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].set_title("Slice Count Distribution")
        axes[0, 0].set_xlabel("Number of Slices")
        axes[0, 0].set_ylabel("Frequency")
        
        axes[0, 1].hist(df['shape_h'], bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].set_title("Height Distribution")
        axes[0, 1].set_xlabel("Height (pixels)")
        axes[0, 1].set_ylabel("Frequency")
        
        axes[0, 2].hist(df['shape_w'], bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 2].set_title("Width Distribution")
        axes[0, 2].set_xlabel("Width (pixels)")
        axes[0, 2].set_ylabel("Frequency")
        
        # Spacing distributions
        if 'spacing_x' in df.columns:
            axes[1, 0].hist(df['spacing_x'], bins=30, edgecolor='black', alpha=0.7, color='purple')
            axes[1, 0].set_title("X Spacing Distribution")
            axes[1, 0].set_xlabel("Spacing (mm)")
            axes[1, 0].set_ylabel("Frequency")
            
            axes[1, 1].hist(df['spacing_y'], bins=30, edgecolor='black', alpha=0.7, color='brown')
            axes[1, 1].set_title("Y Spacing Distribution")
            axes[1, 1].set_xlabel("Spacing (mm)")
            axes[1, 1].set_ylabel("Frequency")
            
            axes[1, 2].hist(df['spacing_z'], bins=30, edgecolor='black', alpha=0.7, color='red')
            axes[1, 2].set_title("Z Spacing (Slice Thickness)")
            axes[1, 2].set_xlabel("Spacing (mm)")
            axes[1, 2].set_ylabel("Frequency")
        
        # HU distributions
        axes[2, 0].hist(df['hu_min'], bins=30, edgecolor='black', alpha=0.7, color='cyan')
        axes[2, 0].set_title("Minimum HU Values")
        axes[2, 0].set_xlabel("HU")
        axes[2, 0].set_ylabel("Frequency")
        
        axes[2, 1].hist(df['hu_max'], bins=30, edgecolor='black', alpha=0.7, color='magenta')
        axes[2, 1].set_title("Maximum HU Values")
        axes[2, 1].set_xlabel("HU")
        axes[2, 1].set_ylabel("Frequency")
        
        # HU percentiles box plot
        hu_percentiles = df[['hu_p01', 'hu_p25', 'hu_median', 'hu_p75', 'hu_p99']]
        axes[2, 2].boxplot([hu_percentiles[col].dropna() for col in hu_percentiles.columns],
                           labels=['1%', '25%', '50%', '75%', '99%'])
        axes[2, 2].set_title("HU Percentiles Distribution")
        axes[2, 2].set_xlabel("Percentile")
        axes[2, 2].set_ylabel("HU")
        
        plt.tight_layout()
        plot_path = self.output_dir / "inspection_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Plots saved to: {plot_path}")
        
        # Additional plot: Anisotropy
        if 'anisotropy_z_to_xy' in df.columns:
            fig2, ax = plt.subplots(figsize=(8, 6))
            ax.hist(df['anisotropy_z_to_xy'], bins=30, edgecolor='black', alpha=0.7, color='teal')
            ax.axvline(1.0, color='red', linestyle='--', label='Isotropic (1.0)')
            ax.set_title("Anisotropy Ratio (Z/XY)")
            ax.set_xlabel("Ratio")
            ax.set_ylabel("Frequency")
            ax.legend()
            plt.tight_layout()
            aniso_path = self.output_dir / "anisotropy_plot.png"
            plt.savefig(aniso_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Anisotropy plot saved to: {aniso_path}")
    
    def save_results(self, df: pd.DataFrame):
        """Save detailed results"""
        print("\n💾 Saving detailed results...")
        
        # Save volume statistics
        stats_path = self.output_dir / "volume_statistics.csv"
        df.to_csv(stats_path, index=False)
        print(f"   ✅ Volume statistics: {stats_path}")
        
        # Save quality issues
        if self.quality_issues:
            issues_path = self.output_dir / "quality_issues.json"
            with open(issues_path, 'w') as f:
                json.dump(self.quality_issues, f, indent=2)
            print(f"   ✅ Quality issues: {issues_path}")
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata_list, f, indent=2, default=str)
        print(f"   ✅ Metadata: {metadata_path}")
        
        # Generate preprocessing config
        config = {
            "hu_clip_range": [float(df['hu_p01'].median()), float(df['hu_p99'].median())],
            "target_spacing": [
                float(df['spacing_x'].median()) if 'spacing_x' in df.columns else 1.0,
                float(df['spacing_y'].median()) if 'spacing_y' in df.columns else 1.0,
                float(df['spacing_z'].median()) if 'spacing_z' in df.columns else 2.0
            ] if 'spacing_x' in df.columns else [1.0, 1.0, 2.0],
            "target_shape": [128, 512, 512],  # [D, H, W]
            "normalization": "z_score_after_clip",
            "notes": "Generated from dataset inspection"
        }
        
        config_path = self.output_dir / "preprocessing_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"   ✅ Preprocessing config: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Inspect CT volume dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to CT data directory")
    parser.add_argument("--format", type=str, default="auto", choices=["auto", "dicom", "nifti", "npy"],
                        help="Data format (auto-detect by default)")
    parser.add_argument("--output_dir", type=str, default="inspection_results",
                        help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to inspect (for quick testing)")
    
    args = parser.parse_args()
    
    # Create inspector
    inspector = CTVolumeInspector(args.data_dir, args.output_dir)
    
    # Run inspection
    inspector.inspect_dataset(format_type=args.format, max_samples=args.max_samples)
    
    # Generate summary
    df = inspector.generate_summary()
    
    if df is not None:
        # Generate plots
        inspector.generate_plots(df)
        
        # Save results
        inspector.save_results(df)
    
    print("\n" + "="*60)
    print("✅ INSPECTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
