import os
import random
import argparse

parser = argparse.ArgumentParser(description='RTFM')
parser.add_argument('--anomaly_dir', default='/scratch/aishaeld/I3D_Feature_Extraction_resnet/custom_features/anomaly')
parser.add_argument('--non_anomaly_dir', default='/scratch/aishaeld/I3D_Feature_Extraction_resnet/custom_features/non_anomaly')

parser.add_argument('--test_file', default='custom_test_list.list')


args = parser.parse_args()

# Directories containing the files
anomaly_dir = args.anomaly_dir

non_anomaly_dir = args.non_anomaly_dir

# Percentage of files to use for training
# train_percent = 0.6

# Output file names
# train_file = args.train_file
test_file = args.test_file

# Get a list of files in the directories
anomaly_files = os.listdir(anomaly_dir)
non_anomaly_files = os.listdir(non_anomaly_dir)

# Shuffle the file lists
random.shuffle(anomaly_files)
random.shuffle(non_anomaly_files)

# Calculate the number of files to use for training
# anomaly_train_count = int(len(anomaly_files) * train_percent)
# non_anomaly_train_count = int(len(non_anomaly_files) * train_percent)

# Write the train files to the output files
# with open(train_file, 'w') as f:
#     for i in range(anomaly_train_count):
#         f.write(os.path.join(anomaly_dir, anomaly_files[i]) + '\n')
#     for i in range(non_anomaly_train_count):
#         f.write(os.path.join(non_anomaly_dir, non_anomaly_files[i]) + '\n')

# Write the test files to the output files
with open(test_file, 'w') as f:
    for i in range(len(anomaly_files)):
        f.write(os.path.join(anomaly_dir, anomaly_files[i]) + '\n')

    for i in range(len(non_anomaly_files)):
        f.write(os.path.join(non_anomaly_dir, non_anomaly_files[i]) + '\n')
        # print( non_anomaly_files[i])

# Print the number of files at each step
# with open('config.txt', 'w') as f:
#     f.write(f'Number of anomaly files: {len(anomaly_files)}\n')
#     f.write(f'Number of non-anomaly files: {len(non_anomaly_files)}\n')
#     f.write(f'Number of anomaly files for training: {anomaly_train_count}\n')
#     f.write(f'Number of non-anomaly files for training: {non_anomaly_train_count}\n')
#     f.write(f'Number of anomaly files for testing: {len(anomaly_files) - anomaly_train_count}\n')
#     f.write(f'Number of non-anomaly files for testing: {len(non_anomaly_files) - non_anomaly_train_count}\n')
