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
from utils import prepare_device, seed_everything
from utils.parse_config import ConfigParser
import yaml
from efficientnet_pytorch import EfficientNet

def train(cfg) -> None:
    logger = cfg.get_logger('train')
    logger.debug(f'Training: {cfg}')
    seed_everything(cfg['seed'])
    # setup data_loader instances
    data_loader = cfg.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = cfg.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(cfg['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = cfg.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = cfg.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


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
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    train(config)
    # model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5).cuda()
    # state_dict = torch.load("E:/Ameme/saved/models/Cassava/0127_211800/checkpoint-epoch30.pth")["state_dict"] # 模型可以保存为pth文件，也可以为pt文件。
    # # create new OrderedDict that does not contain `module.`
    # test = torch.load("E:/Ameme/saved/models/Cassava/0127_211800/checkpoint-epoch30.pth")
    # from collections import OrderedDict
    #
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[6:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
    #     new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    # # load params
    # model.load_state_dict(new_state_dict)