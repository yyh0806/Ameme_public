import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import transforms
from base import DataLoaderBase
from PIL import Image
import os
from data_loader.datasets import CassavaDataset
from data_loader.augmentation import CassavaTransforms
import numpy as np
import pandas as pd


class CassavaDataLoader(DataLoaderBase):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=0):
        train = pd.read_csv(data_dir[:-13] + 'train.csv')
        X_Train, Y_Train = train['image_id'].values, train['label'].values
        transforms = CassavaTransforms()
        self.train_dataset = CassavaDataset(data_dir, X_Train, Y_Train, transforms)
        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers
        }
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers)


class MnistDataLoader(DataLoaderBase):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir: str, batch_size: int, shuffle: bool = True, validation_split: float = 0.0,
                 num_workers=0, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
