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
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer = optimizer

        # configuration to monitor model performance and save best
        self._setup_monitoring(config['trainer'])

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir
        if config['trainer']['visdom']:
            self.viz = Visdom()
            Popen("visdom", shell=True, stdout=PIPE, stderr=PIPE)
            self.viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    def train(self):
        """
        Full training logic
        """
        self.logger.info('Starting training...')
        for epoch in tqdm(range(self.start_epoch, self.epochs + 1)):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            for key, value in log.items():
                if key == 'metrics':
                    log.update({
                        mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                    for i, mtr in enumerate(self.metrics):
                        self.viz.line([value[i]], [epoch], win=mtr[i], update='append')
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

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            not_improved_count = 0
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according
                    # to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(f"Warning: Metric '{self.mnt_metric}' is not found. Model "
                                        "performance monitoring is disabled.")
                    self.mnt_mode = 'off'
                    improved = False
                    not_improved_count = 0

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(f"Validation performance didn\'t improve for {self.early_stop} "
                                     "epochs. Training stops.")
                    break

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
            'monitor_best': self.mnt_best,
            # 'config': self.config
        }
        filename = self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth'
        torch.save(state, filename)
        self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = self.checkpoint_dir / 'model_best.pth'
            torch.save(state, best_path)
            self.logger.info(f'Saving current best: {best_path}')

    def _setup_monitoring(self, config: dict) -> None:
        """
        Configuration to monitor model performance and save best.
        """
        self.epochs = config['epochs']
        self.save_period = config['save_period']
        self.monitor = config.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = math.inf if self.mnt_mode == 'min' else -math.inf
            self.early_stop = config.get('early_stop', math.inf)

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
