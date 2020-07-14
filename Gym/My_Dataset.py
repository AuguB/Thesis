import numpy as np
import torch
from torch.utils.data.dataset import Dataset

class MyDataset(Dataset):
    def __init__(self, data, transform = None, noise = None):
        self.data = data
        self.transform = transform
        self.noise = noise

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        dataPoint = self.data[i]
        if self.transform:
            dataPoint = self.transform(dataPoint)
        return dataPoint