from torch.utils.data import DataLoader
from torch.optim import optimizer
from torch import nn
import torch
from .TrainerConfig import TrainerConfig
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


class Trainer:
    def __init__(self, model, trainLoader: DataLoader, validLoader: DataLoader, device: torch.device,
                 optim: optimizer, criterion: nn.Module, config: TrainerConfig = None):
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

    def train_one_epoch(self):
        self.model.train()

        for samples, targets in tqdm(self.trainLoader):
            samples = samples.to(self.device).float()
            targets = targets.unsqueeze(1).to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(samples)
            loss = self.criterion(outputs, targets)
            self.loss = loss
            self.targets = targets
            self.outputs = outputs
            loss.backward()

            self.optimizer.step()

    def validation(self):
        self.model.eval()
        y_true = torch.Tensor().cuda()
        y_pred = torch.Tensor().cuda()
        for samples, targets in tqdm(self.validLoader):
            with torch.no_grad():
                samples = samples.to(self.device).float()
                targets = targets.unsqueeze(1).to(self.device).float()

                outputs = self.model(samples)
                preds = torch.sigmoid(outputs)
                loss = self.criterion(outputs, targets)
                y_true = torch.cat((y_true, targets))
                y_pred = torch.cat((y_pred, preds))
                self.loss = loss
                self.targets = targets
                self.outputs = outputs

        f1score = f1_score(y_true.cpu(), torch.round(y_pred.cpu()))
        aucscore = roc_auc_score(y_true.cpu(), torch.round(y_pred.cpu()))
        return f1score, aucscore

    def fit(self, epochs: int = 20):
        if self.config is not None and self.config.n_epochs > 0:
            for ep in range(self.config.n_epochs):
                self.train_one_epoch()
                f1score, aucscore = self.validation()
                print(ep, f1score, aucscore)


