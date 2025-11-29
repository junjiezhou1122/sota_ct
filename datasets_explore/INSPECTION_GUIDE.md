# CT Data Inspection Guide

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/junjie/Desktop/reserach/sota_ct/datasets_explore

# Activate your virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements_inspection.txt
```

### 2. Run Inspection

#### For DICOM data:
```bash
python inspect_ct_volumes.py \
    --data_dir /path/to/your/dicom/data \
    --format dicom \
    --output_dir inspection_results
```

#### For NIfTI data:
```bash
python inspect_ct_volumes.py \
    --data_dir /path/to/your/nifti/data \
    --format nifti \
    --output_dir inspection_results
```

#### Auto-detect format:
```bash
python inspect_ct_volumes.py \
    --data_dir /path/to/your/ct/data \
    --format auto
```

#### Quick test (first 10 samples):
```bash
python inspect_ct_volumes.py \
    --data_dir /path/to/your/ct/data \
    --max_samples 10
```

---

## What the Script Does

### Phase 1: File Discovery
- Searches for CT files (DICOM directories, NIfTI files, NumPy arrays)
- Reports total count and file types found

### Phase 2: Volume Loading & Analysis
For each CT volume:
- **Loads** the 3D data
- **Extracts metadata**: spacing, dimensions, scanner info
- **Computes statistics**: HU values, shape, quality metrics
- **Detects issues**: extreme values, small images, missing slices

### Phase 3: Summary Generation
- **Dimension statistics**: min/max/median for D×H×W
- **Spacing analysis**: voxel size in X, Y, Z
- **HU distribution**: percentiles, min/max, mean/std
- **Anisotropy ratio**: Z-spacing vs XY-spacing
- **Scanner metadata**: manufacturers, kernels

### Phase 4: Recommendations
Generates preprocessing suggestions:
- **HU clipping range** (based on 1st-99th percentiles)
- **Target spacing** (based on median)
- **Slice handling strategy** (padding/sliding window)

### Phase 5: Visualization
Creates plots:
- Shape distributions (D, H, W)
- Spacing distributions (X, Y, Z)
- HU value ranges
- Anisotropy ratios

---

## Output Files

All outputs are saved in `inspection_results/` (or your specified `--output_dir`):

### 1. `volume_statistics.csv`
Detailed statistics for each volume:
- Path
- Shape (D, H, W)
- Spacing (X, Y, Z)
- HU statistics (min, max, percentiles)
- Quality flags

**Use for:** Filtering bad samples, understanding data distribution

### 2. `dataset_summary.txt`
Human-readable summary:
- Overall statistics
- Preprocessing recommendations
- Quality issue counts

**Use for:** Quick overview, documentation

### 3. `quality_issues.json`
List of all detected problems:
- Load errors
- Extreme HU values
- Small images
- Missing slices

**Use for:** Data cleaning decisions

### 4. `metadata.json`
Raw metadata from all volumes:
- DICOM tags
- Scanner parameters
- Acquisition details

**Use for:** Advanced analysis, protocol standardization

### 5. `preprocessing_config.json`
Ready-to-use config:
```json
{
  "hu_clip_range": [-1000, 400],
  "target_spacing": [1.5, 1.5, 2.5],
  "target_shape": [128, 512, 512],
  "normalization": "z_score_after_clip"
}
```

**Use for:** Directly in your preprocessing pipeline

### 6. `inspection_plots.png`
9-panel visualization:
- Slice count histogram
- Height/width distributions
- Spacing distributions
- HU value ranges
- Percentile boxplots

### 7. `anisotropy_plot.png`
Shows Z/XY spacing ratio distribution

---

## Interpreting Results

### HU Values

**Normal ranges:**
- Air: -1000
- Lung tissue: -900 to -500
- Fat: -120 to -90
- Water: 0
- Soft tissue: 20 to 70
- Blood: 30 to 45
- Bone: 200 to 3000

**Warning signs:**
- `hu_min < -2000`: Likely artifacts or errors
- `hu_max > 5000`: Metal artifacts or calibration issues
- Very narrow range: Possible windowing already applied

### Spacing

**Typical CT spacing:**
- In-plane (X, Y): 0.5 to 1.0 mm (high-res) or 1.0 to 2.0 mm (standard)
- Slice thickness (Z): 1.0 to 5.0 mm

**Anisotropy ratio (Z/XY):**
- `< 2.0`: Good (nearly isotropic)
- `2.0-4.0`: Acceptable (standard CT)
- `> 4.0`: High anisotropy (may need special handling)

### Slice Count

**Common scenarios:**
- Chest CT: 100-400 slices
- Thin-slice HRCT: 400-800 slices
- Thick-slice screening: 50-150 slices

**If your target is 128 slices:**
- Volumes with 128-256 slices: Can crop/sample directly
- Volumes with < 128 slices: Need padding or interpolation
- Volumes with > 256 slices: Use sliding window or downsampling

---

## Common Issues & Solutions

### Issue: "No CT files found"
**Solutions:**
1. Check `--data_dir` path is correct
2. Verify file permissions
3. Try `--format auto` to detect multiple formats
4. Check if data is nested in subdirectories

### Issue: "SimpleITK not available"
**Solution:**
```bash
pip install SimpleITK
```

### Issue: "Error loading DICOM"
**Possible causes:**
1. Corrupted DICOM files
2. Missing series information
3. Mixed series in same directory

**Solution:**
- Check `quality_issues.json` for specific errors
- Validate DICOM files with `dcmtk` tools

### Issue: Extreme HU values
**Diagnosis:**
- `hu_min < -2000` or `hu_max > 5000`

**Actions:**
1. Check if CT is calibrated correctly
2. Look for metal artifacts
3. Consider aggressive clipping: `[-1200, 600]`

### Issue: High anisotropy (>5)
**Impact:**
- 3D convolutions may not work well
- Slice-by-slice processing might be better

**Solutions:**
1. Resample to isotropic spacing (slower)
2. Use anisotropic kernels in your model
3. Use 2.5D approach (process slices with context)

---

## Next Steps After Inspection

### 1. Based on Summary, Decide:
- **HU normalization**: Clip → Z-score or Min-max?
- **Target spacing**: Resample all to median spacing?
- **Target shape**: 128 or 160 slices?
- **Handling outliers**: Filter out or keep?

### 2. Create Preprocessing Pipeline
Use the generated `preprocessing_config.json` as input to your preprocessing script.

### 3. Data Cleaning
- Filter volumes with quality issues
- Remove corrupted files
- Handle edge cases (too small, too large)

### 4. Document Dataset
- Add inspection results to your paper/documentation
- Report statistics in "Dataset" section
- Justify preprocessing choices based on inspection

---

## Example Workflow

```bash
# 1. Quick test with 10 samples
python inspect_ct_volumes.py --data_dir /data/ct --max_samples 10

# 2. Review results
cat inspection_results/dataset_summary.txt

# 3. Full inspection
python inspect_ct_volumes.py --data_dir /data/ct

# 4. Check for issues
cat inspection_results/quality_issues.json

# 5. Use config in preprocessing
# Copy preprocessing_config.json to your pipeline
cp inspection_results/preprocessing_config.json ../preprocessing/config.json
```

---

## Extending the Script

### Add Custom Quality Checks

Edit `analyze_volume()` method:

```python
# Check for motion artifacts (high std in edges)
edge_std = np.std(volume[:, :10, :])
if edge_std > threshold:
    issues.append("Possible motion artifact")
```

### Add More Metadata

Edit `load_dicom_volume()`:

```python
metadata['patient_age'] = getattr(dcm, 'PatientAge', None)
metadata['study_date'] = getattr(dcm, 'StudyDate', None)
```

### Custom Plots

Add to `generate_plots()`:

```python
# Plot HU histogram
plt.figure()
plt.hist(all_hu_values, bins=100, range=(-1000, 500))
plt.xlabel('HU')
plt.ylabel('Frequency')
plt.title('Global HU Distribution')
plt.savefig(output_dir / 'hu_histogram.png')
```

---

## Tips for Different Data Sources

### RadGenome-ChestCT (Hugging Face)
```python
from huggingface_hub import hf_hub_download

# Download and inspect
data_dir = hf_hub_download(
    repo_id="RadGenome/RadGenome-ChestCT",
    filename="ct_scans/",
    repo_type="dataset",
    local_dir="./data"
)

# Run inspection
python inspect_ct_volumes.py --data_dir ./data/ct_scans
```

### LIDC-IDRI
- Each case has multiple series
- Need to select CT series (not segmentations)
- Use SeriesDescription to filter

### Private Hospital Data
- May have PHI in DICOM tags → be careful with metadata.json
- Check for consistent protocols
- Validate against IRB requirements

---

## Performance Notes

### Speed
- **Small dataset (<100 scans)**: 1-5 minutes
- **Large dataset (1000+ scans)**: 30-60 minutes
- **Sampling**: Uses 10M voxels max per volume for HU stats

### Memory
- Loads one volume at a time
- Peak memory ≈ largest volume size
- Typical: 2-4 GB RAM sufficient

### Optimization
For very large datasets:
1. Use `--max_samples` for testing
2. Run in parallel on HPC cluster
3. Process in batches with separate output dirs

---

## Troubleshooting

### Script crashes on specific file
1. Check `quality_issues.json` for error details
2. Try loading that file manually:
```python
import SimpleITK as sitk
image = sitk.ReadImage("path/to/problematic/file")
```

### Memory error
1. Reduce sampling if you modified the code
2. Process fewer samples at once
3. Increase system RAM

### Plots not generated
1. Check if matplotlib is installed
2. Verify DISPLAY environment variable (if SSH)
3. Use `Agg` backend for headless systems:
```python
import matplotlib
matplotlib.use('Agg')
```

---

## Contact

For issues or questions about this inspection tool:
- Check the code comments in `inspect_ct_volumes.py`
- Review error messages in terminal output
- Consult `quality_issues.json` for specific problems

Good luck with your CT data analysis! 🏥🔬
