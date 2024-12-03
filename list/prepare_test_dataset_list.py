import os
import argparse

def collect_feature_paths(directory):
    """
    Collect .npy feature file paths from 'anomaly' and 'non_anomaly' subfolders.
    """
    subfolders = ['anomaly', 'non_anomaly']
    feature_paths = []
    
    for subfolder in subfolders:
        folder_path = os.path.join(directory, subfolder)
        if os.path.exists(folder_path):
            print(f"Checking folder: {folder_path}")
            for file in os.listdir(folder_path):
                # Include only .npy files
                if file.endswith(".npy"):
                    feature_paths.append(os.path.join(folder_path, file))
        else:
            print(f"Warning: Subfolder '{subfolder}' does not exist in {directory}. Skipping.")
    
    return feature_paths

def save_list(file_path, data):
    """
    Save the list of file paths to a .list file.
    """
    with open(file_path, 'w') as f:
        for item in data:
            f.write(f"{item}\n")

def main():
    parser = argparse.ArgumentParser(description="Prepare test dataset list")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                        help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the .list file.")
    args = parser.parse_args()
    
    # Collect feature file paths
    feature_paths = collect_feature_paths(args.dataset_dir)
    
    if not feature_paths:
        print("No feature files found. Check your dataset directory and parameters.")
        return
    
    # Save to .list file
    os.makedirs(args.output_dir, exist_ok=True)
    output_list_path = os.path.join(args.output_dir, "test.list")
    save_list(output_list_path, feature_paths)
    
    # Verify and display results
    total_files = len(feature_paths)
    print(f"Total number of feature files: {total_files}")
    print(f"Feature list saved to {output_list_path}")

if __name__ == "__main__":
    main()
