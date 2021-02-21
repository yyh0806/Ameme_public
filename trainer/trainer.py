import numpy as np
import torch
from torchvision.utils import make_grid
from utils import inf_loop, MetricTracker, mixup_data, mix_criterion, VisdomLinePlotter, cutmix_data
from base import TrainerBase
from logger.logger import setup_logging
from visdom import Visdom
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


class Trainer(TrainerBase):
    def __init__(self, model, epoch, criterion, metrics, optimizer,  device, data_loader, valid_data_loader=None,
                 lr_scheduler=None, len_epoch=None, checkpoint=None, st_process=None, st_stop=False):
        super().__init__(model, criterion, metrics, optimizer, epoch, checkpoint, st_stop=st_stop)
        self.scaler = GradScaler()
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            self.len_epoch = len(self.data_loader)
        else:
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.processBar = st_process
        self.st_stop = st_stop
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=None)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=None)

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            if self.st_stop:
                break
            with autocast():
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.train_metrics.update('loss', loss.item())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                for i, met in enumerate(self.metrics):
                    self.train_metrics.update(met.__name__, met(output, target))

                if batch_idx % self.log_step == 0:
                    self.processBar.progress(batch_idx / self.len_epoch)
                if batch_idx == self.len_epoch:
                    break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            if type(self.lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.lr_scheduler.step(log["val_loss"])
            else:
                self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch: int) -> dict:
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.valid_metrics.update('loss', loss.item())

                for met in self.metrics:
                    self.valid_metrics.update(met.__name__, met(output, target))
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
