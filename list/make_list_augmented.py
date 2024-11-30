import os
import argparse

# Argument parser setup
parser = argparse.ArgumentParser(description='Create training and testing lists with accurate file counts.')
parser.add_argument('--train_val_dir', default='/home/aishaeld/scratch/dataset/features/train_val', help='Training and Validation features directory')
parser.add_argument('--test_dir', default='/home/aishaeld/scratch/dataset/features/test', help='Testing features directory')
parser.add_argument('--train_file', default='training_list_with_augmented.list', help='Output file for the training list')
parser.add_argument('--test_file', default='testing_list.list', help='Output file for the testing list')
parser.add_argument('--config_file', default='config.txt', help='Output file for the configuration summary')

args = parser.parse_args()

# Directories
train_val_dir = args.train_val_dir
test_dir = args.test_dir

# Output files
train_file = args.train_file
test_file = args.test_file
config_file = args.config_file

# Helper function to get all files from a directory
def get_files_in_folder(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

# Define specific folders for counting
anomaly_train_dir = os.path.join(train_val_dir, "anomaly")
non_anomaly_train_dir = os.path.join(train_val_dir, "non_anomaly")
anomaly_augmented_train_dir = [os.path.join(train_val_dir, "anomaly_cropped"), os.path.join(train_val_dir, "anomaly_augmented")]
non_anomaly_augmented_train_dir = [os.path.join(train_val_dir, "non_anomaly_cropped"), os.path.join(train_val_dir, "non_anomaly_augmented")]

anomaly_test_dir = os.path.join(test_dir, "anomaly")
non_anomaly_test_dir = os.path.join(test_dir, "non_anomaly")

# Count training files
anomaly_train_files = get_files_in_folder(anomaly_train_dir)
non_anomaly_train_files = get_files_in_folder(non_anomaly_train_dir)
anomaly_augmented_files = []
non_anomaly_augmented_files = []

for folder in anomaly_augmented_train_dir:
    anomaly_augmented_files += get_files_in_folder(folder)

for folder in non_anomaly_augmented_train_dir:
    non_anomaly_augmented_files += get_files_in_folder(folder)

# Count testing files
anomaly_test_files = get_files_in_folder(anomaly_test_dir)
non_anomaly_test_files = get_files_in_folder(non_anomaly_test_dir)

# Write training list to file
with open(train_file, 'w') as f:
    for file in anomaly_train_files + non_anomaly_train_files + anomaly_augmented_files + non_anomaly_augmented_files:
        f.write(file + '\n')

# Write testing list to file
with open(test_file, 'w') as f:
    for file in anomaly_test_files + non_anomaly_test_files:
        f.write(file + '\n')

# Print and write counts to a config file
with open(config_file, 'w') as f:
    # Training counts
    f.write("Training Data Counts:\n")
    f.write(f" - Number of anomaly training files: {len(anomaly_train_files)}\n")
    f.write(f" - Number of non-anomaly training files: {len(non_anomaly_train_files)}\n")
    f.write(f" - Number of augmented anomaly training files: {len(anomaly_augmented_files)}\n")
    f.write(f" - Number of augmented non-anomaly training files: {len(non_anomaly_augmented_files)}\n\n")

    # Testing counts
    f.write("Testing Data Counts:\n")
    f.write(f" - Number of anomaly testing files: {len(anomaly_test_files)}\n")
    f.write(f" - Number of non-anomaly testing files: {len(non_anomaly_test_files)}\n")

print("Training Data Counts:")
print(f" - Number of anomaly training files: {len(anomaly_train_files)}")
print(f" - Number of non-anomaly training files: {len(non_anomaly_train_files)}")
print(f" - Number of augmented anomaly training files: {len(anomaly_augmented_files)}")
print(f" - Number of augmented non-anomaly training files: {len(non_anomaly_augmented_files)}")

print("\nTesting Data Counts:")
print(f" - Number of anomaly testing files: {len(anomaly_test_files)}")
print(f" - Number of non-anomaly testing files: {len(non_anomaly_test_files)}")

print(f"\nTraining list saved to: {train_file}")
print(f"Testing list saved to: {test_file}")
print(f"Configuration file saved to: {config_file}")
