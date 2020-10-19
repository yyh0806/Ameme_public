# @Time : 2020/9/5 10:54
# @Author : yangyuhui
# @Site : 
# @File : main.py
# @Software: PyCharm

from config import cfg
import logging
import sys
import os
import streamlit as st
import torch
from log import logger
import models.craft_model as module_model
import models.loss as module_loss
import models.metric as module_metric
import models.optimizer as module_optimizer
import models.scheduler as module_scheduler
import dataset.augmentation as module_aug
import dataset.dataloader as module_data
from train.trainer import Trainer
from log.logger import setup_logger, setup_logging
from utils import get_instance


class Ameme:

    def __init__(self, config):
        setup_logging(config)
        self.logger = setup_logger(self, config.TRAIN.VERBOSE)
        self.config = config

    def train(self):
        config = self.config
        device_id = config.TRAIN.DEVICE
        self.logger.debug("Building models")
        model = get_instance(module_model, cfg.MODEL.NAME, cfg.MODEL[cfg.MODEL.NAME])
        device_ids = list(range(torch.cuda.device_count()))
        self.logger.debug(f'Using device {device_id} of {device_ids}')
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device)
        model = model.to(device)
        torch.backends.cudnn.benchmark = True
        self.logger.debug("Building optimizer")
        optimizer = get_instance(module_optimizer, cfg.OPTIMIZER.NAME, cfg.OPTIMIZER.PARAMS)
        self.logger.debug("Building lr scheduler")
        lr_scheduler = get_instance(module_scheduler, cfg.LR_SCHEDULER.NAME, cfg.LR_SCHEDULER.PARAMS, optimizer)
        self.logger.debug("Getting augmentations")
        transforms = get_instance(module_aug, cfg.AUGMENTATION.NAME, cfg.AUGMENTATION.PARAMS)
        self.logger.debug("Getting dataloader instance")
        data_loader = get_instance(module_data, cfg.DATASET.NAME, cfg.DATASET.PARAMS, transforms)
        valid_data_loader = data_loader.split_validation()
        self.logger.debug("Getting loss")
        loss = get_instance(module_loss, cfg.LOSS.NAME, cfg.LOSS.PARAMS)
        loss.to(device)
        self.logger.debug("Getting metrics")
        metrics = [getattr(module_metric, met) for met in cfg.METRICS]
        self.logger.debug("Initialising trainer")
        trainer = Trainer(model, loss, metrics, optimizer,
                          config=config,
                          device=device,
                          trainLoader=data_loader,
                          validLoader=valid_data_loader,
                          lr_scheduler=lr_scheduler)
        trainer.train()
        self.logger.debug("Finished")

Ameme(cfg).train()