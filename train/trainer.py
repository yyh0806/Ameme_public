from torch.utils.data import DataLoader
from torch.optim import optimizer
from torch import nn
import torch
from tqdm import tqdm
from base import BaseTrainer


class Trainer(BaseTrainer):

    def __init__(self, model, loss: nn.Module, metrics, optimizer: optimizer, config, device: torch.device,
                 trainLoader: DataLoader, validLoader: DataLoader, lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, config, device)

        self.trainLoader = trainLoader
        self.validLoader = validLoader
        self.lr_scheduler = lr_scheduler
        self.is_validation = self.validLoader is not None

    def _train_epoch(self, epoch):
        self.model.train()

        losses_comb = AverageMeter('loss_comb')
        losses_bce = AverageMeter('loss_bce')
        losses_dice = AverageMeter('loss_dice')
        losses_iou = AverageMeter('loss_iou')
        metrics = [AverageMeter(m.__name__) for m in self.metrics]

        for batch_idx, (samples, targets) in tqdm(self.trainLoader):
            samples = samples.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(samples)

            loss_dict = self.loss(outputs, targets)
            loss = loss_dict['loss']
            bce = loss_dict.get('bce', torch.tensor([0]))
            dice = loss_dict.get('dice', torch.tensor([0]))
            iou = loss_dict.get('iou', torch.tensor([0]))

            loss.backward()
            self.optimizer.step()

            losses_comb.update(loss.item(), samples.size(0))
            losses_bce.update(bce.item(),   samples.size(0))
            losses_dice.update(dice.item(), samples.size(0))
            losses_iou.update(iou.item(), samples.size(0))

        del samples
        del targets
        del outputs
        torch.cuda.empty_cache()

        if self.is_validation:
            self._valid_epoch(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def _valid_epoch(self):
        self.model.eval()
        with torch.no_grad():
            for samples, targets in tqdm(self.validLoader):
                samples = samples.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(samples)
                loss = self.loss(outputs, targets)

        del samples
        del targets
        del outputs
        torch.cuda.empty_cache()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

