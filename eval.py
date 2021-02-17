import os
import random
from typing import Any, List, Tuple, Dict
from types import ModuleType
import argparse
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as module_optimizer
import torch.optim.lr_scheduler as module_scheduler

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from logger.logger import setup_logging
from utils import prepare_device, seed_everything, MyEnsemble
from utils.parse_config import ConfigParser
import yaml
from utils import inf_loop, MetricTracker
from tqdm import tqdm


class Eval:

    def __init__(self, models, criterion, metrics, config, device, valid_data_loader, logger):
        self.models = models
        self.criterion = criterion
        self.config = config
        self.device = device
        self.metrics = metrics
        self.valid_data_loader = valid_data_loader
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=None)
        self.cfg = ConfigParser(config)
        self.logger = logger

    def _ensemble(self):
        model = MyEnsemble(self.models)
        return model

    def eval(self):
        model = self._ensemble()
        model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(self.valid_data_loader)):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.criterion(output, target)
                self.valid_metrics.update('loss', loss.item())
                for met in self.metrics:
                    self.valid_metrics.update(met.__name__, met(output, target))

            for key, value in self.valid_metrics.result().items():
                self.logger.info(f'{str(key):15s}: {value}')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="experiments/cassava_config.yml", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    cfg = ConfigParser.from_args(args, options)

    logger = cfg.get_logger('eval')
    logger.debug(f'eval: {cfg}')
    seed_everything(cfg['seed'])
    # setup data_loader instances
    data_loader = cfg.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg['n_gpu'])
    # build model architecture, then print to console
    eval_models = cfg.config['eval']
    models = []
    for model_dict in eval_models:
        for model_name, load_path in model_dict.items():
            model = cfg.init_model(model_name, cfg.config['arch']['args']['num_classes'])
            model.load_state_dict(torch.load(load_path)['state_dict'])
            models.append(model.to(device))

    # get function handles of loss and metrics
    criterion = getattr(module_loss, cfg.config['loss'])
    metrics = [getattr(module_metric, met) for met in cfg.config['metrics']]

    eval = Eval(models, criterion, metrics, config=cfg.config, device=device, valid_data_loader=valid_data_loader,
                logger=logger)

    eval.eval()
