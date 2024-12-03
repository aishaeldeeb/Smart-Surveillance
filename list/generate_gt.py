import numpy as np
import os
import argparse
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='RTFM')

parser.add_argument('--list_file', required=True, 
                    help="Path to the input .list file containing feature file paths.")
parser.add_argument('--temporal_root', default='/home/aishaeld/scratch/RTFM/list/mat_files', 
                    help="Path to the directory containing temporal .mat files.")

args = parser.parse_args()

# Generate output file name based on the input list file
list_file_name = os.path.basename(args.list_file)
list_file_dir = os.path.dirname(args.list_file)
output_file = os.path.join(list_file_dir, f"{os.path.splitext(list_file_name)[0]}_gt.npy")

# Load the list of feature files
file_list = list(open(args.list_file))
temporal_root = args.temporal_root
mat_name_list = os.listdir(temporal_root)

gt = []
for file in file_list:
    file = file.strip('\n')  # Clean up the path
    features = np.load(file, allow_pickle=True)
    print(f"Loaded features from {file}: {features.shape}")
    num_frame = features.shape[0] * 16

    split_file = file.split('/')[-1].split('_')[0]
    mat_prefix = '_x264.mat'
    mat_file = split_file + mat_prefix

    count = 0
    if 'non_anomaly' in file:  # Normal case
        print(file, " is a non_anomaly")
        for i in range(0, num_frame):
            gt.append(0.0)
            count += 1
    else:  # Anomaly case
        print(file, " is an anomaly")
        for i in range(0, num_frame):
            gt.append(1.0)
            count += 1
        if mat_file in mat_name_list:
            second_event = False
            annots = loadmat(os.path.join(temporal_root, mat_file))
            annots_idx = annots['Annotation_file']['Anno'].tolist()

            start_idx = annots_idx[0][0][0][0]
            end_idx = annots_idx[0][0][0][1]

            if len(annots_idx[0][0]) == 2:
                second_event = True

            # Process annotation indices
            if not second_event:
                for i in range(0, start_idx):
                    gt.append(0.0)
                    count += 1
                if not (end_idx + 1) > num_frame:
                    for i in range(start_idx, end_idx + 1):
                        gt.append(1.0)
                        count += 1
                    for i in range(end_idx + 1, num_frame):
                        gt.append(0.0)
                        count += 1
                else:
                    for i in range(start_idx, end_idx):
                        gt.append(1.0)
                        count += 1
            else:
                start_idx_2 = annots_idx[0][0][1][0]
                end_idx_2 = annots_idx[0][0][1][1]
                for i in range(0, start_idx):
                    gt.append(0.0)
                    count += 1
                for i in range(start_idx, end_idx + 1):
                    gt.append(1.0)
                    count += 1
                for i in range(end_idx + 1, start_idx_2):
                    gt.append(0.0)
                    count += 1
                if not (end_idx_2 + 1) > num_frame:
                    for i in range(start_idx_2, end_idx_2 + 1):
                        gt.append(1.0)
                        count += 1
                    for i in range(end_idx_2 + 1, num_frame):
                        gt.append(0.0)
                        count += 1
                else:
                    for i in range(start_idx_2, end_idx_2):
                        gt.append(1.0)
                        count += 1
                if count != num_frame:
                    print(annots_idx)
                    print(num_frame)
                    print(count)
                    print(end_idx_2 + 1)

    if count != num_frame:
        print(file)
        print('Num of frames is not correct!!')
        exit(1)

# Save the generated ground truth
gt = np.array(gt, dtype=float)
np.save(output_file, gt)

print(f"Ground truth saved to {output_file}")
print(f"Total frames: {len(gt)}")
