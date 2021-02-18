import argparse
import collections
import torch
import logging
from trainer import Trainer
from data_loader.data_loaders import *
from model.model import *
from loss.loss import *
from torch.optim import *
from torch.optim.lr_scheduler import *
from model.metric import *
from logger.logger import setup_logging
from utils import prepare_device, seed_everything
from config import cfg
import streamlit as st
import numpy as np
import pandas as pd


def train(config) -> None:
    setup_logging('train')
    logger = logging.getLogger()
    logger.info(f'Training: {config}')
    seed_everything(config['SEED'])
    # setup data_loader instances
    data_loader = eval(config["DATA_LOADER"]["TYPE"])(**config["DATA_LOADER"]["ARGS"])
    valid_data_loader = data_loader.split_validation()
    # build model architecture, then print to console
    model = eval(config["MODEL"]["TYPE"])(**config["MODEL"]["ARGS"])
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['N_GPU'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = eval(config['LOSS'])
    metrics = [eval(met) for met in config['METRICS']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = eval(config["OPTIMIZER"]["TYPE"])(trainable_params, **config["OPTIMIZER"]["ARGS"])
    lr_scheduler = eval(config["LR_SCHEDULER"]["TYPE"])(optimizer, **config["LR_SCHEDULER"]["ARGS"])

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
    st.title('Ameme')
    sidebar_title = st.sidebar.title("Train")
    loss_selections = ("nll_loss", "CrossEntropyLoss", "FocalCosineLoss", "BiTemperedLoss")
    loss_selectBox = st.sidebar.selectbox("Loss", loss_selections)
    optimizer_selections = ("ts1", "test2")
    optimizer_selectBox = st.sidebar.selectbox("optimizer", optimizer_selections)
    # train(cfg)
