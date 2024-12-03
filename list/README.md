# Generate mat files

python generate_mat_files.py --csv_file /home/aishaeld/scratch/dataset/videos/annotations.csv --output_dir /home/aishaeld/scratch/RTFM/list/mat_files

prepare dataset list:
## Features
- Collect `.npy` feature file paths from a dataset directory.
- Split the dataset into training and validation sets based on a specified ratio.
- Save training and validation file paths into organized directories.

## Requirements
- Python 3.6+
- NumPy

## Usage:
python prepare_train_val_dataset_list.py --dataset_dir /home/aishaeld/scratch/dataset/features/train_val --include_augmented --output_dir /scratch/aishaeld/RTFM/list

* Without augmenetation


Example:
python prepare_train_val_dataset_list.py --dataset_dir /home/aishaeld/scratch/dataset/features/train_val --output_dir /scratch/aishaeld/RTFM/list

 python prepare_test_dataset_list.py --dataset_dir /home/aishaeld/scratch/dataset/features/test --output_dir /scratch/aishaeld/RTFM/list




# Generate gorund truths:
1. test list: python generate_gt.py --list_file /home/aishaeld/scratch/RTFM/list/test.list
2. val list: python generate_gt.py --list_file /home/aishaeld/scratch/RTFM/list/val.list