import argparse
import collections
import torch
import logging
from trainer import Trainer
from data_loader.data_loaders import *
from model.models import *
from loss import *
from optimizer.optimizers import *
from scheduler.schedulers import *
from model.metric import *
from logger.logger import setup_logging
from utils import prepare_device, seed_everything
from config import cfg


def train(config) -> None:
    setup_logging('train')
    logger = logging.getLogger()
    logger.info(f'Training: {config}')
    seed_everything(config['SEED'])
    # setup data_loader instances
    data_loader = eval(config["DATA_LOADER"]["TYPE"])(**config["DATA_LOADER"]["ARGS"])
    valid_data_loader = data_loader.split_validation()
    # build model architecture, then print to console
    model = create_model((config["MODEL"]["TYPE"]))(**config["MODEL"]["ARGS"])
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['N_GPU'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = eval(config['LOSS']).to(device)
    metrics = [eval(met) for met in config['METRICS']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = create_optimizer(config["OPTIMIZER"]["TYPE"])(**config["OPTIMIZER"]["ARGS"], model=model)
    lr_scheduler, num_epochs = create_scheduler(config["LR_SCHEDULER"]["TYPE"])(**config["LR_SCHEDULER"]["ARGS"],
                                                                                optimizer=optimizer)
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


if __name__ == "__main__":
    cfg.merge_from_file("experiments/config.yml")
    cfg.freeze()

    train(cfg)
