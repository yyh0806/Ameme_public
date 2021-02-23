import numpy as np
import torch
from torchvision.utils import make_grid
from utils import inf_loop, MetricTracker, mixup_data, mix_criterion, VisdomLinePlotter, cutmix_data
from base import TrainerBase
from logger.logger import setup_logging
from visdom import Visdom
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import pandas as pd


class Trainer(TrainerBase):
    def __init__(self, model, epoch, criterion, metrics, optimizer, device, data_loader, valid_data_loader=None,
                 lr_scheduler=None, len_epoch=None, checkpoint=None, sts=[]):  # sts=[stop, st_empty, save_dir]
        super().__init__(model, criterion, metrics, optimizer, epoch, checkpoint, save_dir=sts[2], st_stop=sts[0])
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

        self.st_empty = sts[1]
        self.st_container = self.st_empty.beta_container()
        self.lossChart = self.st_container.line_chart()
        self.processBar = self.st_container.progress(0)
        self.epochResult = self.st_container.table()
        self.train_idx = 0

        self.log_step = 100
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
                    self.lossChart.add_rows(pd.DataFrame(self.train_metrics.result(), index=[self.train_idx]))
                    self.train_idx = self.train_idx + 1
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

        st_res = log.copy()
        self.epochResult.add_rows(pd.DataFrame(st_res, index=[epoch]))
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
