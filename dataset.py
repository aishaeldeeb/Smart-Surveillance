New! Keyboard shortcuts … Drive keyboard shortcuts have been updated to give you first-letters navigation
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
        self.n_anomaly = args.n_anomaly
        self.n_non_anomaly = args.n_non_anomaly

        if (mode=="train"):
            self.list_file = args.train_list
            # print(f"train_list: {args.train_list}")
            # print(f"train_list length: {len(args.train_list)}")


        elif (mode=="val"):
            self.list_file = args.val_list
            # print(f"validation_list: {args.val_list}")
            # print(f"validation_list length: {len(args.val_list)}")


        elif (mode=="test"):
            self.list_file = args.test_list
            # print(f"test_list: {args.test_list}")
            # print(f"test_list length: {len(args.test_list)}")



        self.tranform = transform
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    # def _parse_list(self):
        
    #     self.list = list(open(self.list_file))

    #     if(self.mode=="train"):
    #         if self.is_normal:
    #             # last number of non-anomaly videos
    #             self.list = self.list[self.n_non_anomaly:]
    #         else:
    #             # first number of anomaly videos
    #             self.list = self.list[:self.n_anomaly]

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
        file_paths = list(open(self.list_file))
        if self.mode == "train":
            if self.is_normal:
                self.list = file_paths[self.n_non_anomaly:]
            else:
                self.list = file_paths[:self.n_anomaly]
        else:
            self.list = file_paths


    def __getitem__(self, index):

        label = self.get_label(index)  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)
        # features = torch.tensor(features, dtype=torch.float32)
        # print(f"features: {features}")

        if self.tranform is not None:
            features = self.tranform(features)


        # print("Before padding:", features.shape)
        # features = pad_sequence([torch.tensor(f) for f in features], batch_first=True)
        # print("After padding:", features.shape)

        if self.mode == "train":
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)
            # features = torch.tensor(features, dtype=torch.float32)

            return divided_features, label
        
        else:
            return features, label

    # def collate_fn(self, data):
    #     # data is a list of tuples
    #     features, labels = zip(*data)
    #     features = [torch.tensor(f, dtype=torch.float32) for f in features]  # Convert to tensors
   
    #     features = pad_sequence(features, batch_first=True)
    #     labels = torch.tensor(labels)
    #     return features, labels
    
    # def collate_fn(self, data):
    #     # data is a list of tuples
    #     features, labels = zip(*data)
    #     features = [torch.tensor(f, dtype=torch.float32) for f in features]  # Convert to tensors
    #     max_length = max([f.shape[0] for f in features])  # Find the maximum sequence length

    #     # Pad sequences manually
    #     padded_features = []
    #     for f in features:
    #         padding_length = max_length - f.shape[0]
    #         padded_f = torch.cat([f, torch.zeros(padding_length, *f.shape[1:])], dim=0)
    #         padded_features.append(padded_f)

    #     padded_features = torch.stack(padded_features)  # Convert the list to a tensor
    #     labels = torch.tensor(labels)
    #     return padded_features, labels

    def get_label(self, index):

        directory = os.path.dirname(self.list[index].strip('\n'))

        label = 0.0 if "non_anomaly" in directory else 1.0
        # print(f"directory: {directory}")
        # print(f"Label: {label}")
        return torch.tensor(label)

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame