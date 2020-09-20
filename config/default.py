# @Time : 2020/9/5 10:11
# @Author : yangyuhui
# @Site : 
# @File : default.py
# @Software: PyCharm
import os
from yacs.config import CfgNode as CN

_C = CN()

_C.TRAIN = CN()
_C.TRAIN.NUM_WORKERS = 0
_C.TRAIN.MAX_EPOCHS = 25
_C.TRAIN.LR = 0.001

_C.DATASET = CN()
_C.DATASET.BATCH_SIZE = 16

_C.SIDEBAR = CN()
_C.SIDEBAR.TITLE = "AMEME"
_C.SIDEBAR.SUBHEADER_DATAPATH = "Ameme/dataset"

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
