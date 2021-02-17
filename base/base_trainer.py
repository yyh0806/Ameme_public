import logging
from abc import abstractmethod
from pathlib import Path
import math

from subprocess import Popen, PIPE
import yaml
import torch
from tqdm import tqdm
from visdom import Visdom


class TrainerBase:
    """
    Base class for all trainers
    """
    def __init__(self, model, criterion, metrics, optimizer, config):
        self.config = config
        self.logger = logging.getLogger("trainer")

        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer

        self.start_epoch = 1
        self.epochs = config["TRAINER"]["epochs"]
        self.checkpoint_dir = config.SAVE_DIR
        if config.RESUME is not None:
            self._resume_checkpoint(config.RESUME)

    def train(self):
        self.logger.info('Starting training...')
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            log.update(result)

            for key, value in log.items():
                if key == 'metrics':
                    log.update({
                        mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({
                        'val_' + mtr.__name__: value[i] for
                        i, mtr in enumerate(self.metrics)
                    })
                else:
                    log[key] = value

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info(f'{str(key):15s}: {value}')

            best = False

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)

    @abstractmethod
    def _train_epoch(self, epoch: int) -> dict:
        """
        Training logic for an epoch.
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch: int, save_best: bool = False) -> None:
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        filename = self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth'
        torch.save(state, filename)
        self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = self.checkpoint_dir / 'model_best.pth'
            torch.save(state, best_path)
            self.logger.info(f'Saving current best: {best_path}')

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
