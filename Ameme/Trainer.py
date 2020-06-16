from torch.utils.data import DataLoader
from torch.optim import optimizer
from .Meters import *
from torch import nn
import torch
from .TrainerConfig import TrainerConfig
from tqdm import tqdm


class Trainer:

    def __init__(self, model, trainLoader: DataLoader, validLoader: DataLoader, device: torch.device,
                 optim: optimizer, criterion: nn.Module, meters=[], config: TrainerConfig = None):
        self.model = model
        self.trainLoader = trainLoader
        self.validLoader = validLoader
        self.device = device
        self.config = config
        self.optimizer = optim
        self.criterion = criterion.to(self.device)

        self.batchSize = config.batch_size
        self.loss = 0
        self.targets = None
        self.outputs = None

        self._meters = meters

    @property
    def meters(self):
        return self._meters

    @meters.setter
    def meters(self, value):
        self._meters = value

    def train_one_epoch(self):
        self.model.train()

        res_meters = []

        for samples, targets in tqdm(self.trainLoader):
            samples = samples.to(self.device).float()
            targets = targets.to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(samples)
            loss = self.criterion(outputs, targets)
            self.loss = loss
            self.targets = targets
            self.outputs = outputs
            loss.backward()

            self.optimizer.step()

            for meter in self._meters:
                meter.update(self)
                res_meters.append(meter)

        return res_meters

    def validation(self):
        self.model.eval()

        res_meters = []

        for samples, targets in tqdm(self.validLoader):
            with torch.no_grad():
                samples = samples.to(self.device).float()
                targets = targets.to(self.device).float

                outputs = self.model(samples)
                loss = self.criterion(outputs, targets)

                self.loss = loss
                self.targets = targets
                self.outputs = outputs

                for meter in self._meters:
                    meter.update(self)
                    res_meters.append(meter)

        return res_meters

    def fit(self, epochs: int = 20):
        if self.config is not None and self.config.n_epochs > 0:
            for ep in range(self.config.n_epochs):
                res_meters = self.train_one_epoch()
                for meter in res_meters:
                    print(meter.value)
                res_meters = self.validation()


