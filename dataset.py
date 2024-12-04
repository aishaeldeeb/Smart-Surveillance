import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.FloatTensor')
import os
from torch.nn.utils.rnn import pad_sequence

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, mode="train" ):
        
        self.is_normal = is_normal
        self.mode = mode

        if (mode=="train"):
            self.list_file = args.train_list

        elif (mode=="val"):
            self.list_file = args.val_list

        elif (mode=="test"):
            self.list_file = args.test_list

        self.tranform = transform
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def collate_fn(self, batch):
        # batch is a list of tuples: [(features_1, label_1), (features_2, label_2), ...]
        features, labels = zip(*batch)

        # Pad sequences manually
        max_length = max([f.shape[0] for f in features])  # Find the maximum sequence length
        padded_features = []
        for f in features:
            padding_length = max_length - f.shape[0]
            padded_f = np.pad(f, ((0, padding_length), (0, 0), (0, 0)))  # Pad feature sequences
            padded_features.append(padded_f)

        # Convert the list to a tensor
        padded_features = torch.stack([torch.from_numpy(f) for f in padded_features])
        labels = torch.tensor(labels)

        return padded_features, labels

    def _parse_list(self):
        file_paths = list(open(self.list_file))  # Read all file paths
        self.list = []

        # Separate anomalies and non-anomalies dynamically
        anomalies = []
        non_anomalies = []

        for path in file_paths:
            if "anomaly" in path and not "non_anomaly" in path:  # Check if the path indicates an anomaly
                anomalies.append(path.strip('\n'))
            elif "non_anomaly" in path:  # Check if the path indicates a non-anomaly
                non_anomalies.append(path.strip('\n'))

        if self.mode == "train":
            if self.is_normal:
                # Use all non-anomaly samples for training if normal data is required
                self.list = non_anomalies
            else:
                # Use all anomaly samples for training if anomalies are required
                self.list = anomalies
        else:
            # For validation/testing, include both anomalies and non-anomalies
            self.list = anomalies + non_anomalies

    def __getitem__(self, index):

        label = self.get_label(index)  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)


        if self.mode == "train":
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label
        
        else:
            return features, label

    def get_label(self, index):
        directory = os.path.dirname(self.list[index].strip('\n'))

        label = 0.0 if "non_anomaly" in directory else 1.0

        return torch.tensor(label)

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame