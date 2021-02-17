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
    """
    Responsible for training loop and validation.
    """

    def __init__(self, model, criterion, metrics, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None, scaler=GradScaler()):
        super().__init__(model, criterion, metrics, optimizer, config)
        self.scaler = scaler
        self.config = config
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
        self.log_step = int(np.sqrt(data_loader.batch_size))
        if self.config['trainer']['visdom']:
            self.plotter = VisdomLinePlotter(self.viz)

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=None)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=None)

    def _train_epoch(self, epoch: int) -> dict:
        """
        Training logic for an epoch

        Returns
        -------
        dict
            Dictionary containing results for the epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(tqdm(self.data_loader)):
            with autocast():
                data, target = data.to(self.device), target.to(self.device)
                if self.config['trainer']['visdom']:
                    self.viz.images(data, nrow=6, win=1, opts={'title': 'data'})

                self.optimizer.zero_grad()
                output = self.model(data)
                r_mixup = np.random.rand(1)
                r_cutmix = np.random.rand(1)
                if self.config['mixup']['use'] and r_mixup < self.config['mixup']['prob']:
                    data, targets_a, targets_b, lam = mixup_data(data, target, self.config['mixup']['alpha'])
                    if self.config['trainer']['visdom']:
                        self.viz.images(data, nrow=8, win=11, opts={'title': 'mixup'})
                    loss = mix_criterion(self.criterion, output, targets_a, targets_b, lam)
                elif self.config['cutmix']['use'] and r_cutmix < self.config['cutmix']['prob']:
                    data, targets_a, targets_b, lam = cutmix_data(data, target, self.config['cutmix']['alpha'])
                    if self.config['trainer']['visdom']:
                        self.viz.images(data, nrow=8, win=12, opts={'title': 'mixup'})
                    loss = mix_criterion(self.criterion, output, targets_a, targets_b, lam)
                else:
                    loss = self.criterion(output, target)
                self.train_metrics.update('loss', loss.item())
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                for i, met in enumerate(self.metrics):
                    self.train_metrics.update(met.__name__, met(output, target))

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item()))
                if batch_idx == self.len_epoch:
                    break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
            if self.config['trainer']['visdom']:
                self.plotter.plot('loss', 'train', 'train_loss', epoch, log["loss"])
                self.plotter.plot('loss', 'val', 'val_loss', epoch, log["val_loss"])
        if self.lr_scheduler is not None:
            if type(self.lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.lr_scheduler.step(log["val_loss"])
            else:
                self.lr_scheduler.step()
        if self.config['trainer']['visdom']:
            self.viz.text(str(log), win=3)
        return log

    def _valid_epoch(self, epoch: int) -> dict:
        """
        Validate after training an epoch

        Returns
        -------
        dict
            Contains keys 'val_loss' and 'val_metrics'.
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.valid_data_loader)):
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
