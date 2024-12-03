# Smart Surveillance: Customized RTFM for Video Anomaly Detection

## Overview
This repository is a customized version of the original RTFM implementation, tailored for the Smart Surveillance project. It trains neural network models for video anomaly detection using pre-extracted I3D features. Key modifications include:
- Introduction of new data augmentation techniques.
- Enhanced preprocessing pipeline.
- Integration with a specific dataset structure.

---

## Changes Made

### `train.py`
- **Augmentation Strategy**:
  - Replaced 10-crop testing with:
    - Center cropping.
    - Flipping.
    - Brightness level adjustments.
  - Optimized for smaller datasets and computational efficiency.
- **Logging Enhancements**:
  - Detailed logs include augmentation details for better tracking.

### `test.py`
- **Testing Consistency**:
  - Updated to align with the new augmentation techniques in `train.py`.
- **Enhanced Metrics Collection**:
  - Added metrics to analyze anomaly detection performance.

### `dataset.py`
- **Augmentation Integration**:
  - Added support for center cropping, flipping, and brightness adjustments during data loading.
- **Preprocessing Adjustments**:
  - Improved handling of datasets with pre-extracted I3D features.

### `model.py`
- **Architecture Refinements**:
  - Adjusted to support new input variations from the updated augmentation pipeline.

### `config.py` and `option.py`
- **Custom Parameters**:
  - New options to toggle augmentation methods.
  - Default configurations updated for the Smart Surveillance project.

---

## Dataset and Features Structure

Your dataset must be organized as follows, with pre-extracted I3D features:

```bash
dataset/
├── features/
│   ├── train_val/
│   │   ├── anomaly/
│   │   ├── anomaly_augmented/
│   │   ├── anomaly_cropped/
│   │   ├── non_anomaly/
│   │   ├── non_anomaly_augmented/
│   │   ├── non_anomaly_cropped/
│   └── test/
│       ├── anomaly/
│       ├── anomaly_augmented/
│       ├── anomaly_cropped/
│       ├── non_anomaly/
│       ├── non_anomaly_cropped/
└── videos/ # Mirrors the features structure
```

For I3D feature extraction, refer to **[I3D Feature Extraction with ResNet](#https://github.com/aishaeldeeb/I3D_Feature_Extraction/tree/main)**.

---

## Included Scripts

### Job Scripts
The `job_scripts/` folder contains SLURM scripts for automated training and testing on HPC environments.
