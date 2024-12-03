import os
import random
import argparse
import numpy as np

def collect_feature_paths(directory, include_augmented):
    """
    Collect paths to feature files based on whether to include augmented data.
    """
    subfolders = ['anomaly', 'non_anomaly']
    if include_augmented:
        subfolders += ['anomaly_cropped', 'non_anomaly_cropped', 
                       'anomaly_augmented', 'non_anomaly_augmented']
    
    feature_paths = []
    for subfolder in subfolders:
        folder_path = os.path.join(directory, subfolder)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.npy'):  # Ensure only .npy files are collected
                    feature_paths.append(os.path.join(folder_path, file))
    return feature_paths

def split_feature_paths(feature_paths, train_ratio=0.8):
    """
    Shuffle and split feature file paths into training and validation sets.
    """
    random.shuffle(feature_paths)
    total_files = len(feature_paths)
    
    # Calculate exact split
    train_count = int(total_files * train_ratio)
    
    train_features = feature_paths[:train_count]
    val_features = feature_paths[train_count:]
    
    return train_features, val_features

def save_feature_paths(file_path, feature_paths):
    """
    Save the list of feature file paths to a .list file.
    """
    with open(file_path, 'w') as f:
        for path in feature_paths:
            f.write(f"{path}\n")

def main():
    parser = argparse.ArgumentParser(description="Prepare train and validation feature file lists")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                        help="Path to the dataset directory.")
    parser.add_argument("--include_augmented", action="store_true", 
                        help="Include augmented data.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the .list files.")
    args = parser.parse_args()
    
    # Collect feature paths
    feature_paths = collect_feature_paths(args.dataset_dir, args.include_augmented)
    
    if not feature_paths:
        print("No feature files found. Check your dataset directory and parameters.")
        return
    
    # Split into training and validation
    train_features, val_features = split_feature_paths(feature_paths)
    
    # Save to .list files
    os.makedirs(args.output_dir, exist_ok=True)

    if args.include_augmented:
        augmentation = "with_aug"
    else:
        augmentation = "without_aug"

    train_list_path = os.path.join(args.output_dir, f"train_{augmentation}.list")
    val_list_path = os.path.join(args.output_dir, f"val_{augmentation}.list")
    
    save_feature_paths(train_list_path, train_features)
    save_feature_paths(val_list_path, val_features)
    
    # Verify and display results
    total_files = len(feature_paths)
    train_files = len(train_features)
    val_files = len(val_features)
    
    print(f"Total number of files: {total_files}")
    print(f"Training features: {train_files} files ({(train_files / total_files) * 100:.2f}%)")
    print(f"Validation features: {val_files} files ({(val_files / total_files) * 100:.2f}%)")
    print(f"Training feature paths saved to {train_list_path}")
    print(f"Validation feature paths saved to {val_list_path}")

if __name__ == "__main__":
    main()
