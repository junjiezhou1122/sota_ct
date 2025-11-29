# RadGenome-ChestCT Dataset Structure Summary

This document provides a summary of the RadGenome-ChestCT dataset structure from Hugging Face.

## Dataset Overview

The RadGenome-ChestCT dataset contains chest CT scans with associated reports and annotations. The dataset is organized into three main sections:

1. **Training Data** (`dataset/train_preprocessed/`)
2. **Validation Data** (`dataset/valid_preprocessed/`)
3. **Metadata and Reports** (`dataset/radgenome_files/`)

## Dataset Structure

### Training Data (`dataset/train_preprocessed/`)
- Contains 16,242 training cases
- Each case is organized in a hierarchical structure:
  ```
  train_preprocessed/
  ├── train_1/
  │   └── train_1a/
  │       └── train_1_a_1.nii.gz
  ├── train_2/
  │   └── train_2a/
  │       └── train_2_a_1.nii.gz
  ...
  ```
- Each case contains NIfTI files (.nii.gz) with the CT scan data
- Naming convention: `train_[ID]_[letter]_[number].nii.gz`

### Validation Data (`dataset/valid_preprocessed/`)
- Contains 1,000 validation cases
- Similar structure to training data:
  ```
  valid_preprocessed/
  ├── valid_1/
  │   └── valid_1a/
  │       └── valid_1_a_1.nii.gz
  ├── valid_2/
  │   └── valid_2a/
  │       └── valid_2_a_1.nii.gz
  ...
  ```
- Naming convention: `valid_[ID]_[letter]_[number].nii.gz`

### Metadata and Reports (`dataset/radgenome_files/`)
Contains CSV files with various types of annotations and reports:

#### Training Metadata:
- `train_case_disorders.csv` - Disorders for each training case
- `train_region_report.csv` - Regional reports for training cases
- `train_vqa_abnormality.csv` - Visual Question Answering data for abnormalities
- `train_vqa_location.csv` - VQA data for locations
- `train_vqa_presence.csv` - VQA data for presence/absence
- `train_vqa_size.csv` - VQA data for sizes

#### Validation Metadata:
- `validation_case_disorders.csv` - Disorders for each validation case
- `validation_region_report.csv` - Regional reports for validation cases
- `validation_vqa_abnormality.csv` - VQA data for abnormalities
- `validation_vqa_location.csv` - VQA data for locations
- `validation_vqa_presence.csv` - VQA data for presence/absence
- `validation_vqa_size.csv` - VQA data for sizes

## File Statistics

- **Total files**: 25,693
- **Total directories**: 17,243
- **File extensions**:
  - `.nii.gz`: 25,693 files (CT scan data)
  - `.csv`: 12 files (metadata and reports)
  - `.md`: 1 file (README)
  - `.json`: 1 file (configuration)

## Dataset Access

To explore this dataset without downloading:
1. Use the Hugging Face web interface at https://huggingface.co/datasets/RadGenome/RadGenome-ChestCT
2. Use the `explore_dataset_structure.py` script created in this repository
3. Use the Hugging Face CLI: `huggingface-cli repo ls-files RadGenome/RadGenome-ChestCT --repo-type dataset`

## Notes

- The dataset contains preprocessed CT scans in NIfTI format
- Each case may have multiple NIfTI files (indicated by the letter in the filename)
- The VQA (Visual Question Answering) files suggest this dataset can be used for medical VQA tasks
- Regional reports provide detailed information about different anatomical regions
- The dataset is well-structured with clear separation between training and validation sets