import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from utils import inf_loop, MetricTracker, mixup_data, mix_criterion, VisdomLinePlotter, cutmix_data
from base import TrainerBase
from logger.logger import setup_logging
from visdom import Visdom
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import torch.nn.functional as F


class KDTrainer(TrainerBase):
    def __init__(self, s_model, t_model, epoch, criterion, metrics, optimizer, device, data_loader, valid_data_loader=None,
                 lr_scheduler=None, len_epoch=None, checkpoint=None, sts=[]):  # sts=[stop, st_empty, save_dir]
        super().__init__(s_model, criterion, metrics, optimizer, epoch, checkpoint, save_dir=sts[2], st_stop=sts[0])
        self.scaler = GradScaler()
        self.device = device
        self.s_model = self.model
        self.s_model = self.s_model.to(device)
        self.t_model = t_model
        self.t_model = self.t_model.to(device)
        self.kd_criterion = nn.KLDivLoss(size_average=False)
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
        outputs = []
        targets = []
        for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
            if self.st_stop:
                break
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output, loss = self._calculate_loss(data, target)
            outputs.append(output.sigmoid().detach().cpu().numpy())
            targets.append(target.cpu().numpy())
            self.train_metrics.update('loss', loss.item())
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.log_step == 0:
                self.lossChart.add_rows(pd.DataFrame(self.train_metrics.result(), index=[self.train_idx]))
                self.train_idx = self.train_idx + 1
                self.processBar.progress(batch_idx / self.len_epoch)
            if batch_idx == self.len_epoch:
                break
        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)
        for i, met in enumerate(self.metrics):
            self.train_metrics.update(met.__name__, met(outputs, targets))
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
        self.logger.info(log)
        return log

    def _valid_epoch(self, epoch: int) -> dict:
        self.model.eval()
        self.valid_metrics.reset()
        outputs = []
        targets = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output, loss = self._calculate_loss(data, target)
                outputs.append(output.sigmoid().detach().cpu().numpy())
                targets.append(target.cpu().numpy())
                self.valid_metrics.update('loss', loss.item())
            outputs = np.concatenate(outputs)
            targets = np.concatenate(targets)
            for met in self.metrics:
                self.valid_metrics.update(met.__name__, met(outputs, targets))
        return self.valid_metrics.result()

    def _kd_loss(self, out_s, out_t, target):
        alpha = 0.5
        T = 4
        loss = self.criterion(out_s, target)
        batch_size = target.shape[0]
        s_max = F.log_softmax(out_s / T, dim=1)
        t_max = F.softmax(out_t / T, dim=1)
        loss_kd = self.kd_criterion(s_max, t_max) / batch_size
        loss = (1 - alpha) * loss + alpha * T * T * loss_kd
        return loss

    def _calculate_loss(self, data, target):
        out_s = self.s_model(data)
        out_t = self.t_model(data)
        loss = self._kd_loss(out_s, out_t, target)
        return out_s, loss
