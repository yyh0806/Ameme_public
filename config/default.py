# @Time : 2020/9/5 10:11
# @Author : yangyuhui
# @Site : 
# @File : default.py
# @Software: PyCharm
import os
from yacs.config import CfgNode as CN

_C = CN()

#models
_C.MODEL = CN()
_C.MODEL.NAME = "FPN"
#EffNet
_C.MODEL.FPN = CN()
_C.MODEL.FPN.activation = None
_C.MODEL.FPN.decoder_merge_policy = "cat"
_C.MODEL.FPN.encoder_name = "efficientnet-b5"
_C.MODEL.FPN.encoder_weights = "imagenet"
_C.MODEL.FPN.classes = 4
_C.MODEL.FPN.in_channels = 1


#augmentation
_C.AUGMENTATION = CN()
_C.AUGMENTATION.NAME = "HeavyCropTransforms"
_C.AUGMENTATION.PARAMS = CN()
_C.AUGMENTATION.PARAMS.height = 256
_C.AUGMENTATION.PARAMS.width = 416
#train
_C.TRAIN = CN()
_C.TRAIN.NUM_WORKERS = 0
_C.TRAIN.MAX_EPOCHS = 25
_C.TRAIN.LR = 0.001
_C.TRAIN.SAVE_PERIOD = 10
_C.TRAIN.VERBOSE = 0
_C.TRAIN.DEVICE = 0
#dataset
_C.DATASET = CN()
_C.DATASET.NAME = "SteelSegPseudoDataLoader"
_C.DATASET.PARAMS = CN()
_C.DATASET.PARAMS.alpha = -0.15
_C.DATASET.PARAMS.batch_size = 8
_C.DATASET.PARAMS.data_dir = "E:/Kaggle/severstal-steel-defect-detection-master/data/raw/severstal-steel-defect-detection"
_C.DATASET.PARAMS.nworkers = 0
_C.DATASET.PARAMS.shuffle = True
_C.DATASET.PARAMS.validation_split = 0.2
#loss
_C.LOSS = CN()
_C.LOSS.NAME = "BCEDiceLoss"
_C.LOSS.PARAMS = CN()
_C.LOSS.PARAMS.bce_weight = 0.6
_C.LOSS.PARAMS.dice_weight = 0.4
#metircs
_C.METRICS = ["dice_0", "dice_1", "dice_2", "dice_3", "dice_mean"]
#optimizer
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = "RAdam"
_C.OPTIMIZER.PARAMS = CN()
_C.OPTIMIZER.PARAMS.lr = 0.001
_C.OPTIMIZER.PARAMS.weight_decay = 0.0002
_C.OPTIMIZER.DECODER = CN()
_C.OPTIMIZER.DECODER.lr = 0.003
_C.OPTIMIZER.DECODER.weight_decay = 0.0003
_C.OPTIMIZER.ENCODER = CN()
_C.OPTIMIZER.ENCODER.lr = 7.0e-05
_C.OPTIMIZER.ENCODER.weight_decay = 3.0e-05
#lr_scheduler
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.NAME = "CosineAnnealingScheduler"
_C.LR_SCHEDULER.PARAMS = CN()
_C.LR_SCHEDULER.PARAMS.n_epochs = 301
_C.LR_SCHEDULER.PARAMS.start_anneal = 30
#sidebar
_C.SIDEBAR = CN()
_C.SIDEBAR.TITLE = "AMEME"
_C.SIDEBAR.SUBHEADER_DATAPATH = "./dataset"

#logging
_C.LOGGING = CN()


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
