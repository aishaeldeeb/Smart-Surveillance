# Smart Surveillance Training System

This repository implements a framwork to train an anomaly detection model for smart surveillance system applications, leveraging weakly-supervised learning techniques. Building on the work from the [RTFM: Robust Temporal Feature Magnitude Learning](https://github.com/tianyu0207/RTFM), the system detects anomalous events by training a the RTFM model on augmented and annotated video data.

Key Features in this Project:
- **Data Augmentation**: Implements custom augmentation procedures, focusing on cropping anomaly and non-anomaly data around the subject of interest.
- **Feature Extraction**: Uses pre-trained models to extract features from video data.
- **Model Training**: Trains a model for anomaly detection using the augmented and feature-extracted data.
- **Ground-Truth Data**: Annotates video data and generates ground truth for model training and evaluation.

## Installation Instructions

1. Clone this repository:

```bash
git clone https://github.com/aishaeldeeb/Smart_Surveillance.git
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Clone external repositories for feature extraction and augmentation:

[Video Augmentation](https://github.com/aishaeldeeb/Videos-Augmentation)
[Feature Extraction Repo](https://github.com/aishaeldeeb/I3D_Feature_Extraction_resnet)


### Data Augmentation Process

This repository uses manual cropping techniques to focus on the subject of interest in both anomaly and non-anomaly videos. Augmentation is applied using scripts from the [Videos-Augmentation repository](https://github.com/aishaeldeeb/Videos-Augmentation).

**Manual Cropping**: Crop the subject of interest in both anomalous and non-anomalous video frames.

**Augmentation**: After cropping, various augmentations (flipping, rotation, etc.) are applied to original videos to enhance the model's ability to generalize.


### Feature Extraction

We use the pre-trained I3D model, based on ResNet, for extracting temporal features from videos. This is done using the repository [I3D_Feature_Extraction_resnet](https://github.com/aishaeldeeb/I3D_Feature_Extraction_resnet).


### Annotation of Data

Annotations are generated based on video segments for training purposes. The format for annotation files is CSV:
```csv
video_name,start_annotation,end_annotation,event_name
armed_001.mp4,179,719,armed
```
To convert these annotations into MAT files for processing, use the script:

```bash
cd list
python generate_mat_files.py --csv_file /path/to/annotations.csv --output_dir /path/to/mat_files
```

### List Generation

This step involves generating a list of video paths for both the training - validation, and test datasets. 

1. To generate `train.list` and `val.list`, use:

```bash
cd list
python prepare_train_val_dataset_list.py --dataset_dir /path/to/train_val/dataset --output_dir path/to/output/directory
```

2. To generate `test.list`, use:


```bash
cd list
python prepare_test_dataset_list.py --dataset_dir /path/to/train_val/dataset --output_dir path/to/output/directory
```

These lists will be used during model training and evaluation.

#### Training with Augmented Data
To streamline the process of comparing the performance between augmented and non_augmented data training, you can include the augmented data in the training - validation, and test datasets, use:

```bash
python prepare_train_val_dataset_list.py --dataset_dir /path/to/train_val/dataset --include_augmented --include_augmented --output_dir path/to/output/directory
```

### Ground Truth Mask Extraction

Extract ground truth masks from the MAT files. This is necessary for model training and evaluation, as it provides the expected anomaly locations.

1. Run the ground truth extration for the validation dataset:
```bash
cd list
python generate_gt.py --list_file /path/to/test.list
```

2. Run the ground truth extration for the test dataset:

```bash
cd list
python generate_gt.py --list_file /path/to/val.list
```

### Model Training

Before running training, ensure you adjust the configuration parameters in `option.py` for batch size, learning rate, data paths:

Run the training script with:
```bash
python main.py \
    --train-list <path_to_train_list> \
    --val-list <path_to_val_list> \
    --test-list <path_to_test_list> \
    --train-list <path_to_train_list> \
    --val_gt <path_to_val_ground_truth>
```

Note: Adjust the input paths and hyperparameters according to your dataset and desired configurations. The training script will use the feature-extracted and augmented data for anomaly detection.

# Changes to Code

This repository makes several modifications to the original [RTFM repository](https://github.com/tianyu0207/RTFM) to enhance its functionality for anomaly detection:


---

## Changes to `main.py`:

1. **Loss Function Enhancements**:
   - Introduced custom loss functions:
     - **Sparsity Loss**: Encourages feature sparsity in anomaly detection.
     - **Smoothness Loss**: Promotes temporal consistency in predictions.
     - **RTFM Loss**: Combines classification loss and feature-based margin constraints.

2. **Training Logic Simplification**:
   - Consolidated the training loop within `main.py`, eliminating the dependency on external configuration files.
   - Removed periodic testing and checkpointing to streamline the codebase.

3. **Visualization Removal**:
   - Commented out visualization components (`viz.plot_lines`) to focus on core training.

4. **Static Dataset and Device Setup**:
   - Replaced dynamic dataset and device configuration (`args`, `config`) with direct definitions in the code.

5. **Streamlined Imports**:
   - Reduced imports to essential modules, focusing on `torch` utilities and custom loss functions.

6. **Custom Regularization Terms**:
   - Added sparsity and smoothness regularizers to improve anomaly detection robustness.

---

## Changes to `dataset.py`:

1. **Generalized Dataset Handling**:
   - Replaced hardcoded dataset-specific logic with dynamic file path parsing (`train_list`, `val_list`, `test_list`) based on `args` and `mode` (train, val, test).

2. **Custom Collate Function**:
   - Introduced a `collate_fn` for DataLoader:
     - Pads sequences to the maximum length in the batch for consistent input size.
     - Converts padded features and labels into tensors.

3. **Dynamic Label Inference**:
   - Labels are derived directly from the file paths, identifying `"non_anomaly"` directories for normal samples.

4. **Extended Modes**:
   - Added support for validation mode (`val`) alongside training and testing.

5. **Enhanced Debugging**:
   - Commented out debugging statements for dataset parsing and file lists, making them optional for troubleshooting.

6. **Flexible Preprocessing**:
   - Retained the original feature segmentation logic (`process_feat`) but added support for additional transformations through the `transform` parameter.

## Using `test.py` instead of test_10crop.py:
1. **Video-Level Evaluation**:
   - Added logic for video-level anomaly detection by aggregating frame-level scores.
   - Classifies a video as anomalous if more than 10% of its frames exceed a threshold.

2. **Dynamic Ground Truth Loading**:
   - Ground truth files (`val_gt` and `test_gt`) are dynamically loaded based on the mode (`val` or `test`).

3. **Expanded Metrics**:
   - Calculated video-level AUC and precision-recall curves, in addition to frame-level metrics.

4. **Improved Output Organization**:
   - Removed visualization (`viz`) and saved results in structured directories (`./output/`), with timestamps.


### Acknowledgments

- This work builds upon the [RTFM: Robust Temporal Feature Magnitude Learning](https://github.com/tianyu0207/RTFM).
