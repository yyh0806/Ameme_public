import argparse
import collections
import torch
import logging
from trainer import Trainer
from data_loader.data_loaders import *
import model.model as model_module
import loss.loss as loss_module
import optimizer.optimizer as optimizer_module
import scheduler.scheduler as scheduler_module
from model.metric import *
from logger.logger import setup_logging
from utils import prepare_device, seed_everything
from config import cfg
import streamlit as st
import numpy as np
import pandas as pd
import inspect
import utils.SessionState as session


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
    st.set_page_config(page_title='Ameme', page_icon=":shark:", layout='centered', initial_sidebar_state='auto')
    cfg.merge_from_file("experiments/config.yml")
    cfg.freeze()

    model_selections = []
    for name, obj in inspect.getmembers(model_module, inspect.isclass):
        if "model.model" in str(obj):
            model_selections.append(str(obj)[20:-2])
    loss_selections = []
    for name, obj in inspect.getmembers(loss_module, inspect.isfunction):
        loss_name = str(obj).split(" ")[1]
        loss_selections.append(loss_name)

    trainer_container = st.sidebar.beta_container()
    trainer_container.title('Trainer')
    model_selectBox = trainer_container.selectbox("Model", model_selections)
    loss_selectBox = trainer_container.selectbox("Loss", loss_selections)

    add_btn = trainer_container.button("确定")

    session_trainer = session.get(trainer_dict={"models": [], "losses": []})

    if add_btn:
        session_trainer.trainer_dict["models"].append(model_selectBox)
        session_trainer.trainer_dict["losses"].append(loss_selectBox)

    trainer_dataFrame = st.dataframe(session_trainer.trainer_dict)

    # optimizer_selections = ("test1", "test2")
    # optimizer_selectBox = optimizer_col.selectbox("optimizer", optimizer_selections)
    # scheduler_selections = ("cosine", "step")
    # scheduler_selectBox = scheduler_col.selectbox("scheduler", scheduler_selections)
