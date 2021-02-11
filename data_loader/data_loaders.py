import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

from base import DataLoaderBase
from PIL import Image
import os
from data_loader.datasets import CassavaDataset
from data_loader.augmentation import CassavaTransforms
import numpy as np
import pandas as pd


class MnistDataLoader(DataLoaderBase):
    """
    MNIST data loading demo using DataLoaderBase
    """

    def __init__(self, transforms, data_dir, batch_size, shuffle, validation_split, nworkers,
                 train=True):
        self.data_dir = data_dir

        self.train_dataset = datasets.MNIST(
            self.data_dir,
            train=train,
            download=True,
            transform=transforms.build_transforms(train=True)
        )
        self.valid_dataset = datasets.MNIST(
            self.data_dir,
            train=False,
            download=True,
            transform=transforms.build_transforms(train=False)
        ) if train else None

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers
        }
        super().__init__(self.train_dataset, shuffle=shuffle, **self.init_kwargs)

    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            return DataLoader(self.valid_dataset, **self.init_kwargs)


class CassavaDataLoader(DataLoaderBase):

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=0):
        # train = pd.read_csv('E:/Ameme/data/cassava-leaf-disease-classification/train.csv')
        train = pd.read_csv(data_dir[:-12] + 'train.csv')
        X_Train, Y_Train = train['image_id'].values, train['label'].values
        transforms = CassavaTransforms()
        self.train_dataset = CassavaDataset(data_dir, X_Train, Y_Train, transforms)
        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': num_workers
        }
        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers)

