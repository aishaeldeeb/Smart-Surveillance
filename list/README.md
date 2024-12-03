This directory contains scripts and resources for preparing datasets and annotations. Follow the steps below to understand and utilize the tools provided.

---

## Contents
1. **`generate_mat_files.py`**: A script for generating `.mat` annotation files.
2. **Dataset List Preparation**: Scripts to convert train, validation, and test datasets directories into list files.
3. **`generate_gt.py`**: A script to generate ground-truth mask files in `.npy`, using the annotations in `.mat` files.

---

## Generating `.mat` Files

Use the `generate_mat_files.py` script to convert a `.csv` annotation file into `.mat` files compatible with the system.

### Command-line Usage:
```bash
python generate_mat_files.py --csv_file <path_to_csv> --output_dir <path_to_output>
```

## Arguments

### Generating `.mat` Files
- `--csv_file`: Path to the input CSV file with annotations.
- `--output_dir`: Directory to save the generated `.mat` files.

---

## Annotation File Format
Following is an example on how the `.csv` annotation file should be structured:

```bash
video_name,start_annotation,end_annotation,event_name,second_event,frame_rate,start_time,end_time,second_start_time,seond_end_time
armed_001.mp4,179,719,armed,0,29.97002997,6,24,0,0
armed_002.mp4,179,839,armed,0,29.97002997,6,28,0,0
armed_003.mp4,149,629,armed,0,29.97002997,5,21,0,0
```

## Preparing Dataset Lists

Scripts are provided to organize `.npy` feature files into training, validation, and test sets.

### Command-line Usage:
```bash
python prepare_list.py --dataset_dir <path_to_dataset> --output_dir <path_to_output> 
```

### Arguments:
--dataset_dir: Root directory containing .npy feature files.
--output_dir: Directory to save the resulting dataset lists.

* Augmented Data:
To include augmented data, add the --include_augmented flag:

```bash
python prepare_list.py --dataset_dir <path_to_dataset> --output_dir <path_to_output> --split_ratio 0.8 --include_augmented
```

## Requirements
Ensure the following dependencies are installed:
- Python 3.6+
- NumPy
