from torch.utils.data import DataLoader
from torch.optim import optimizer
from .metrics import AverageMeter, RocAucMeter
from torch import nn
import torch
from .TrainerConfig import TrainerConfig
from tqdm import tqdm


class Trainer:

    def __init__(self, model, trainLoader: DataLoader, validLoader: DataLoader, device: torch.device,
                 optim: optimizer, criterion: nn.Module, config: TrainerConfig = None, meters: dict = None):
        self.model = model
        self.trainLoader = trainLoader
        self.validLoader = validLoader
        self.device = device
        self.config = config
        self.optimizer = optim
        self.criterion = criterion.to(self.device)
        self.meters = meters

    def train_one_epoch(self):
        self.model.train()

        summary_loss = AverageMeter.AverageMeter()
        final_scores = RocAucMeter.RocAucMeter()

        for samples, targets in tqdm(self.trainLoader):
            samples = samples.to(self.device).float()
            targets = targets.to(self.device).float()
            batchSize = samples.shape[0]

            self.optimizer.zero_grad()
            outputs = self.model(samples)
            loss = self.criterion(outputs, targets)
            loss.backward()

            final_scores.update(targets, outputs)
            summary_loss.update(loss.detach().item(), batchSize)

            self.optimizer.step()

        return summary_loss, final_scores

    def validation(self):
        self.model.eval()

        summary_loss = AverageMeter.AverageMeter()
        final_scores = RocAucMeter.RocAucMeter()

        for samples, targets in tqdm(self.validLoader):
            with torch.no_grad():
                samples = samples.to(self.device).float()
                targets = targets.to(self.device).float
                batch_size = samples.shape[0]
                outputs = self.model(samples)
                loss = self.criterion(outputs, targets)

                summary_loss.update(loss.detach().item(), batch_size)
                final_scores.update(targets, outputs)

        return summary_loss, final_scores

    def fit(self, epochs: int = 20):
        if self.config is not None and self.config.n_epochs > 0:
            for ep in range(self.config.n_epochs):
                summary_loss, final_scores = self.train_one_epoch()
                print("loss:", summary_loss, "score:", final_scores)
                summary_loss, final_scores = self.validation()
                print("loss:", summary_loss, "score:", final_scores)