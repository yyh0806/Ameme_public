import numpy as np
import torch
from torchvision.utils import make_grid

from Ameme.base import TrainerBase, AverageMeter
from Ameme.utils import setup_logger


log = setup_logger(__name__)


class Trainer(TrainerBase):
    """
    Responsible for training loop and validation.
    """

    def __init__(self, model, loss, metrics, optimizer, start_epoch, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, loss, metrics, optimizer, start_epoch, config, device)
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size)) * 8

    def _train_epoch(self, epoch: int) -> dict:
        """
        Training logic for an epoch

        Returns
        -------
        dict
            Dictionary containing results for the epoch.
        """
        self.model.train()

        loss_mtr = AverageMeter('loss')
        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            loss_mtr.update(loss.item(), data.size(0))


        del data
        del target
        del output
        torch.cuda.empty_cache()

        results = {
            'loss': loss_mtr.avg,
            'metrics': [mtr.avg for mtr in metric_mtrs]
        }

        if self.do_validation:
            val_results = self._valid_epoch(epoch)
            results = {**results, **val_results}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return results

    def _log_batch(self, epoch, batch_idx, batch_size, len_data, loss):
        n_samples = batch_size * len_data
        n_complete = batch_idx * batch_size
        percent = 100.0 * batch_idx / len_data
        msg = f'Train Epoch: {epoch} [{n_complete}/{n_samples} ({percent:.0f}%)] Loss: {loss:.6f}'
        log.debug(msg)

    def _eval_metrics(self, output, target):
        with torch.no_grad():
            for metric in self.metrics:
                value = metric(output, target)
                yield value

    def _valid_epoch(self, epoch: int) -> dict:
        """
        Validate after training an epoch

        Returns
        -------
        dict
            Contains keys 'val_loss' and 'val_metrics'.
        """
        self.model.eval()
        loss_mtr = AverageMeter('loss')
        metric_mtrs = [AverageMeter(m.__name__) for m in self.metrics]
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss(output, target)
                loss_mtr.update(loss.item(), data.size(0))
                for mtr, value in zip(metric_mtrs, self._eval_metrics(output, target)):
                    mtr.update(value, data.size(0))

        del data
        del target
        del output
        torch.cuda.empty_cache()

        return {
            'val_loss': loss_mtr.avg,
            'val_metrics': [mtr.avg for mtr in metric_mtrs]
        }
