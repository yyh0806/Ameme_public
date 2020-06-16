from glob import glob
from sklearn.model_selection import GroupKFold
import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import sklearn

from Ameme.Trainer import *
from Ameme.TrainerConfig import *
from efficientnet_pytorch import EfficientNet
from Ameme.losses.LabelSmoothing import LabelSmoothing
from Ameme.Meters import *



SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)

dataset = []

for label, kind in enumerate(['Cover', 'JMiPOD', 'JUNIWARD', 'UERD']):
    for path in glob('F:/data/Cover/*.jpg'):
        dataset.append({
            'kind': kind,
            'image_name': path.split('\\')[-1],
            'label': label
        })

random.shuffle(dataset)
dataset = pd.DataFrame(dataset)

gkf = GroupKFold(n_splits=5)

dataset.loc[:, 'fold'] = 0
for fold_number, (train_index, val_index) in enumerate(
        gkf.split(X=dataset.index, y=dataset['label'], groups=dataset['image_name'])):
    dataset.loc[dataset.iloc[val_index].index, 'fold'] = fold_number


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(height=512, width=512, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)


def get_valid_transforms():
    return A.Compose([
        A.Resize(height=512, width=512, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)


DATA_ROOT_PATH = 'F:/data'


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class DatasetRetriever(Dataset):

    def __init__(self, kinds, image_names, labels, transforms=None):
        super().__init__()
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        kind, image_name, label = self.kinds[index], self.image_names[index], self.labels[index]
        image = cv2.imread(f'{DATA_ROOT_PATH}/{kind}/{image_name}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        if self.transforms:
            sample = {'image': image}
            sample = self.transforms(**sample)
            image = sample['image']

        target = onehot(4, label)
        return image, target

    def __len__(self) -> int:
        return self.image_names.shape[0]

    def get_labels(self):
        return list(self.labels)


fold_number = 0

train_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold'] != fold_number].kind.values,
    image_names=dataset[dataset['fold'] != fold_number].image_name.values,
    labels=dataset[dataset['fold'] != fold_number].label.values,
    transforms=get_train_transforms(),
)

validation_dataset = DatasetRetriever(
    kinds=dataset[dataset['fold'] == fold_number].kind.values,
    image_names=dataset[dataset['fold'] == fold_number].image_name.values,
    labels=dataset[dataset['fold'] == fold_number].label.values,
    transforms=get_valid_transforms(),
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=TrainerConfig.batch_size,
    pin_memory=False,
    drop_last=True,
    num_workers=0,
)
val_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=TrainerConfig.batch_size,
    num_workers=0,
    shuffle=False,
    sampler=SequentialSampler(validation_dataset),
    pin_memory=False,
)
device = torch.device('cpu')
net = EfficientNet.from_pretrained('efficientnet-b2')
net._fc = nn.Linear(in_features=1408, out_features=4, bias=True)
optimizer = torch.optim.AdamW(net.parameters(), lr=TrainerConfig.lr)
criterion = LabelSmoothing().to(device)
meters = [AvgLossMeter(), RocAucMeter()]
trainer = Trainer(net, trainLoader=train_loader, validLoader=val_loader, device=device, optim=optimizer
                  , criterion=criterion, meters=meters, config=TrainerConfig)
trainer.fit(20)

