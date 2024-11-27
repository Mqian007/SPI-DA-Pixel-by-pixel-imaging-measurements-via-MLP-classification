import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

class DATA(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.labels = []

        for label in range(11):
            folder_path = os.path.join(self.root_dir, str(label))
            files = os.listdir(folder_path)
            for file in files:
                file_path = os.path.join(folder_path, file)
                data_sample = np.loadtxt(file_path, dtype=np.float32)
                self.data.append(data_sample)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_sample = self.data[idx]

        noise_level = np.random.uniform(0, 0.1)  ### noise level change in here

        noise = np.random.normal(0, noise_level, data_sample.shape)
        data_sample += noise
        data_sample = data_sample[:235]


        if data_sample[0] != 0:           ### Please note that this is not a necessary step, please do it when measuring data.
                                           ##Here again, the measured data is normalized.
            data_sample /= data_sample[0]
        total = np.sum(data_sample)
        if total != 0:
            data_sample = (data_sample / total) * 100


        return torch.tensor(data_sample, dtype=torch.float32), self.labels[idx]


