import argparse
import collections
import torch
import logging
from trainer import Trainer
import data_loader.data_loaders as data_module
import model as model_module
import loss.loss as loss_module
import optimizer.optimizer as optimizer_module
import scheduler.scheduler as scheduler_module
import model.metric as metric_module
from logger.logger import setup_logging
from utils import prepare_device, seed_everything
from config import cfg
import streamlit as st
import numpy as np
import os
import signal
import pandas as pd
import inspect
import utils.SessionState as session
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

if __name__ == "__main__":
    st.set_page_config(page_title='Ameme', page_icon=":shark:", layout='centered', initial_sidebar_state='auto')
    cfg.merge_from_file("experiments/config.yml")
    cfg.freeze()
    data_selections = []
    for name, obj in inspect.getmembers(data_module, inspect.isclass):
        if "data_loader.data_loaders" in str(obj):
            data_selections.append(str(obj)[33:-12])
    model_selections = []
    for name, obj in inspect.getmembers(model_module, inspect.isclass):
        if "model.model" in str(obj):
            model_selections.append(str(obj)[20:-2])
    loss_selections = []
    for name, obj in inspect.getmembers(loss_module, inspect.isfunction):
        loss_name = str(obj).split(" ")[1]
        loss_selections.append(loss_name)
    optimizer_selections = []
    for name, obj in inspect.getmembers(optimizer_module, inspect.isfunction):
        optimizer_name = str(obj).split(" ")[1]
        optimizer_selections.append(optimizer_name)
    scheduler_selections = []
    for name, obj in inspect.getmembers(scheduler_module, inspect.isfunction):
        scheduler_name = str(obj).split(" ")[1]
        scheduler_selections.append(scheduler_name)
    metric_selections = []
    for name, obj in inspect.getmembers(metric_module, inspect.isfunction):
        if name.startswith("_"):
            continue
        metric_selections.append(name)
    trainer_container = st.sidebar.beta_container()
    trainer_container.title('Trainer')
    data_selectBox = trainer_container.selectbox("Data", data_selections)
    model_selectBox = trainer_container.selectbox("Model", model_selections)
    loss_selectBox = trainer_container.selectbox("Loss", loss_selections)
    optimizer_selectBox = trainer_container.selectbox("Optimizer", optimizer_selections)
    scheduler_selectBox = trainer_container.selectbox("Scheduler", scheduler_selections)
    metric_selects = trainer_container.multiselect("Metric", metric_selections, default="accuracy")
    add_btn = trainer_container.button("Add Trainer")

    max_width = 1600
    padding_top = 1
    padding_right = 1
    padding_left = 1
    padding_bottom = 1
    COLOR = (255, 255, 255)
    BACKGROUND_COLOR = (0, 0, 0)

    st.markdown(
        f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: {max_width}px;
            padding-top: {padding_top}rem;
            padding-right: {padding_right}rem;
            padding-left: {padding_left}rem;
            padding-bottom: {padding_bottom}rem;
        }}
        .reportview-container .main {{
            color: {COLOR};
            background-color: {BACKGROUND_COLOR};
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )
    sessions = session.get(
        key=0,
        id=0,
        trainer_params={},
        trainer_dict={"id": [], "dataloader": [], "model": [], "loss": [], "optimizer": [], "scheduler": [],
                      "metrics": []})

    if add_btn:
        sessions.trainer_dict["dataloader"].append(data_selectBox)
        sessions.trainer_dict["model"].append(model_selectBox)
        sessions.trainer_dict["loss"].append(loss_selectBox)
        sessions.trainer_dict["optimizer"].append(optimizer_selectBox)
        sessions.trainer_dict["scheduler"].append(scheduler_selectBox)
        sessions.trainer_dict["metrics"].append(metric_selects)
        sessions.trainer_dict["id"].append(sessions.id)
        sessions.id = sessions.id + 1

    trainer_dataFrame = pd.DataFrame(sessions.trainer_dict)
    gb = GridOptionsBuilder.from_dataframe(trainer_dataFrame)
    gb.configure_selection('multiple', use_checkbox=False, rowMultiSelectWithClick=False,
                           suppressRowDeselection=True)
    gridOptions = gb.build()
    grid_response = AgGrid(
        trainer_dataFrame,
        gridOptions=gridOptions,
        height=180,
        width='100%',
        data_return_mode=DataReturnMode.FILTERED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=True,
    )
    trainer_res = grid_response['selected_rows']

    for idx, trainer in enumerate(trainer_res):
        trainer_id = trainer["id"]
        trainer_dataloader = trainer["dataloader"]
        trainer_model = trainer["model"]
        trainer_optimizer = trainer["optimizer"]
        trainer_loss = trainer["loss"]
        trainer_scheduler = trainer["scheduler"]
        trainer_metrics = trainer["metrics"]

        trainer_params_container = st.beta_container()
        trainer_params_container.header("trainer_"+str(trainer_id))
        model_params_col, optimizer_params_col, scheduler_params_col = trainer_params_container.beta_columns(3)

        # model_params_col.header("model")
        # optimizer_params_col.header("optimizer")
        # scheduler_params_col.header("scheduler")

        model_params_col_ex = model_params_col.beta_expander("model params")
        optimizer_params_col_ex = optimizer_params_col.beta_expander("optimizer params")
        scheduler_params_col_ex = scheduler_params_col.beta_expander("scheduler params")

        model_params = {}
        optimizer_params = {}
        scheduler_params = {}

        # init
        if trainer_id not in sessions.trainer_params.keys():
            sessions.trainer_params[trainer_id] = {}
        params_dict = sessions.trainer_params[trainer_id]

        if "model_params" not in params_dict.keys():
            params_dict["model_params"] = {}
        if "optimizer_params" not in params_dict.keys():
            params_dict["optimizer_params"] = {}
        if "scheduler_params" not in params_dict.keys():
            params_dict["scheduler_params"] = {}
        if "pid" not in params_dict.keys():
            params_dict["pid"] = 0

        model_sig = inspect.signature(eval('model_module.' + trainer_model).__init__)
        for name, param in model_sig.parameters.items():
            if name == "self":
                continue
            if name in list(params_dict["model_params"].keys()):
                param_input = model_params_col_ex.text_input(str(trainer_id) + "_model_" + name,
                                                             value=params_dict["model_params"][name])
            else:
                param_input = model_params_col_ex.text_input(str(trainer_id) + "_model_" + name)
            if param_input:
                if "str" in str(param.annotation):
                    sessions.trainer_params[trainer_id]["model_params"][name] = param_input
                else:
                    sessions.trainer_params[trainer_id]["model_params"][name] = param.annotation(eval(param_input))
        optimizer_sig = inspect.signature(eval('optimizer_module.' + trainer_optimizer))
        for name, param in optimizer_sig.parameters.items():
            if name == "kwargs":
                continue
            if name in list(params_dict["optimizer_params"].keys()):
                param_input = optimizer_params_col_ex.text_input(str(trainer_id) + "_optimizer_" + name,
                                                                 value=params_dict["optimizer_params"][name])
            elif param.default != "<class 'inspect._empty'>":
                param_input = optimizer_params_col_ex.text_input(str(trainer_id) + "_optimizer_" + name,
                                                                 value=param.default)
            else:
                param_input = optimizer_params_col_ex.text_input(str(trainer_id) + "_optimizer_" + name)
            if param_input:
                if "str" in str(param.annotation):
                    sessions.trainer_params[trainer_id]["optimizer_params"][name] = param_input
                else:
                    sessions.trainer_params[trainer_id]["optimizer_params"][name] = param.annotation(eval(param_input))

        scheduler_sig = inspect.signature(eval('scheduler_module.' + trainer_scheduler))
        for name, param in scheduler_sig.parameters.items():
            if name == "kwargs":
                continue
            if name in list(params_dict["scheduler_params"].keys()):
                param_input = scheduler_params_col_ex.text_input(str(trainer_id) + "_scheduler_" + name,
                                                                 value=params_dict["scheduler_params"][name])
            elif param.default != "<class 'inspect._empty'>":
                param_input = scheduler_params_col_ex.text_input(str(trainer_id) + "_scheduler_" + name,
                                                                 value=param.default)
            else:
                param_input = scheduler_params_col_ex.text_input(str(trainer_id) + "_scheduler_" + name)
            if param_input:
                if "str" in str(param.annotation):
                    sessions.trainer_params[trainer_id]["scheduler_params"][name] = param_input
                else:
                    sessions.trainer_params[trainer_id]["scheduler_params"][name] = param.annotation(eval(param_input))

        trainer_params_callback = {"processBar": 0, "train_metric": {}}
        trainer_process = st.beta_expander("train")
        trainer_control_col, trainer_graphic_col = trainer_process.beta_columns((1, 2))
        trainer_seed = trainer_control_col.number_input("train_"+str(trainer_id)+"_seed", min_value=0, max_value=10000, value=86)
        trainer_epoch = trainer_control_col.number_input("train_"+str(trainer_id)+"_epochs", min_value=1, max_value=1000, value=20)
        trainer_save_path = trainer_control_col.text_input("train_"+str(trainer_id)+"_save_path", value=trainer_dataloader + '/' + str(trainer_id) + '/')
        trainer_start_btn = trainer_control_col.button("trainer_" + str(trainer_id) + "_start")
        trainer_stop_btn = trainer_control_col.button("trainer_" + str(trainer_id) + "_stop")
        trainer_chart = trainer_graphic_col.empty()
        # TODO delete config file
        if trainer_start_btn:
            logger = logging.getLogger()
            seed_everything(trainer_seed)
            logger.info(trainer)
            logger.info("seed: {}, save_dir: {}", str(trainer_seed), str(trainer_save_path))
            # data
            data_loader = eval("data_module." + trainer_dataloader + "DataLoader")(**cfg["DATA_LOADER"]["ARGS"])
            valid_data_loader = data_loader.split_validation()
            # model
            model = eval("model_module." + trainer_model)(**sessions.trainer_params[trainer_id]["model_params"])
            # logger.info(model)
            # gpu
            device, device_ids = prepare_device(cfg['N_GPU'])
            model = model.to(device)
            if len(device_ids) > 1:
                model = torch.nn.DataParallel(model, device_ids=device_ids)
            # criterion
            criterion = eval("loss_module." + trainer_loss)
            # metrics
            metrics = [eval("metric_module." + met) for met in eval(trainer_metrics)]

            # optimizer
            optimizer = eval("optimizer_module." + trainer_optimizer)(
                **sessions.trainer_params[trainer_id]["optimizer_params"], params=model.parameters())
            # lr_scheduler
            lr_scheduler = eval("scheduler_module." + trainer_scheduler)(**sessions.trainer_params[trainer_id][
                'scheduler_params'],
                                                                         optimizer=optimizer)
            setup_logging(trainer_save_path)
            trainer = Trainer(model=model,
                              criterion=criterion,
                              metrics=metrics,
                              optimizer=optimizer,
                              epoch=trainer_epoch,
                              device=device,
                              data_loader=data_loader,
                              valid_data_loader=valid_data_loader,
                              lr_scheduler=lr_scheduler,
                              sts=[trainer_stop_btn, trainer_chart, trainer_save_path])
            trainer.train()



