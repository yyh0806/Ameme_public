from torch.utils.data import Dataset
from PIL import Image
import os
import json
import cv2


class CassavaDataset(Dataset):
    def __init__(self, directory, filenames, labels, transforms):
        self.directory = directory
        self.filenames = filenames
        self.transforms = transforms
        self.labels = labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        x = cv2.imread(os.path.join(self.directory, self.filenames[index]))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        if "train" in self.directory:
            if self.transforms is not None:
                return self.transforms.build_transforms(train=True)(image=x)["image"], self.labels[index]
            return x, self.labels[index]
        elif "test" in self.directory:
            if self.transforms is not None:
                return self.transforms.build_transforms(train=False)(image=x)["image"], self.filenames[index]
            return x, self.filenames[index]
