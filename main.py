import argparse
import collections
import torch
import logging
import data_loader.data_loaders as module_data
import loss.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from logger.logger import setup_logging
from utils import prepare_device, seed_everything
from config import cfg


def train(config) -> None:
    setup_logging('train')
    logger = logging.getLogger()
    logger.info(f'Training: {config}')
    seed_everything(config['SEED'])
    # setup data_loader instances
    data_loader = cfg.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # # build model architecture, then print to console
    # model = cfg.init_obj('arch', module_arch)
    # logger.info(model)
    #
    # # prepare for (multi-device) GPU training
    # device, device_ids = prepare_device(cfg['n_gpu'])
    # model = model.to(device)
    # if len(device_ids) > 1:
    #     model = torch.nn.DataParallel(model, device_ids=device_ids)
    #
    # # get function handles of loss and metrics
    # criterion = getattr(module_loss, config['loss'])
    # metrics = [getattr(module_metric, met) for met in config['metrics']]
    #
    # # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    # trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer = cfg.init_obj('optimizer', torch.optimizer, trainable_params)
    #
    # lr_scheduler = cfg.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    #
    # trainer = Trainer(model, criterion, metrics, optimizer,
    #                   config=config,
    #                   device=device,
    #                   data_loader=data_loader,
    #                   valid_data_loader=valid_data_loader,
    #                   lr_scheduler=lr_scheduler)
    #
    # trainer.train()


if __name__ == "__main__":
    cfg.merge_from_file("experiments/config.yml")
    cfg.freeze()

    train(cfg)
