import numpy as np
import os
import glob
import numpy as np
from scipy.io import loadmat
from os import walk
import argparse

parser = argparse.ArgumentParser(description='RTFM')

parser.add_argument('--gt_file', default='/home/aishaeld/scratch/RTFM/list/gt_v2.npy')
parser.add_argument('--test_file', default='test_list_v2.list')

args = parser.parse_args()

# root_path = "/home/aishaeld/scratch/RTFM/list"
# dirs = os.listdir(root_path)


rgb_list_file = args.test_file
temporal_root = "/home/aishaeld/scratch/RTFM/list/mat_files"
mat_name_list = os.listdir(temporal_root)

file_list = list(open(rgb_list_file))
num_frame = 0
gt = []
for file in file_list:

    features = np.load(file.strip('\n'), allow_pickle=True)
    # features = [t.cpu().detach().numpy() for t in features]
    # features = np.array(features, dtype=np.float32)
    # features = np.array([t.numpy() for t in features], dtype=np.float32)
    # features = np.array([t.astype(np.float32) for t in features])
    num_frame = features.shape[0] * 16

    split_file = file.split('/')[-1].split('_')[0]
    mat_prefix = '_x264.mat'
    mat_file = split_file + mat_prefix
 
    count = 0
    # if 'Normal_' in file: # if it's normal
    #     # print('hello')
    #     for i in range(0, num_frame):
    #         gt.append(0.0)
    #         count+=1

    if 'non_anomaly' in file: # if it's normal
        # print('hello')
        print(file, " is a non_anomaly")
        for i in range(0, num_frame):
            gt.append(0.0)
            count+=1

    else: #if it's abnormal # get the name from temporal file
        print(file, " is an anomaly")

        for i in range(0, num_frame):
            gt.append(1.0)
            count+=1
        if mat_file in mat_name_list:
            second_event = False
            annots = loadmat(os.path.join(temporal_root, mat_file))
            annots_idx = annots['Annotation_file']['Anno'].tolist()

            start_idx = annots_idx[0][0][0][0]
            end_idx = annots_idx[0][0][0][1]

            if len(annots_idx[0][0]) == 2:
                second_event = True
                
            # check if there's second events 
            # not needed
            if not second_event:
                for i in range(0, start_idx):
                    gt.append(0.0)
                    count +=1
                if not (end_idx + 1) > num_frame:
                    for i in range(start_idx, end_idx + 1):
                        gt.append(1.0)
                        count += 1
                    for i in range(end_idx+1, num_frame):
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
                for i in range(end_idx+1, start_idx_2):
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
                    print(end_idx_2 +1)


    if count != num_frame:
        print(file)
        print('Num of frames is not correct!!')
        exit(1)



output_file = args.gt_file
gt = np.array(gt, dtype=float)
np.save(output_file, gt)
# for i in range(0, len(gt)):
#     if (gt[i] == 1):
#         print(gt[i])
print(len(gt))





