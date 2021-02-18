from yacs.config import CfgNode as CN

_C = CN()
_C.NAME = "Ameme"
_C.SAVE_DIR = "saved/"
_C.SEED = 8060
_C.N_GPU = 1
_C.RESUME = None

_C.DATA_LOADER = CN()
_C.DATA_LOADER.TYPE = "CassavaDataLoader"
_C.DATA_LOADER.ARGS = CN()
_C.DATA_LOADER.ARGS.data_dir = "E:/Ameme/data/cassava-leaf-disease-classification/train_images/"
_C.DATA_LOADER.ARGS.batch_size = 4
_C.DATA_LOADER.ARGS.shuffle = True
_C.DATA_LOADER.ARGS.validation_split = 0.1
_C.DATA_LOADER.ARGS.num_workers = 0

_C.MODEL = CN()
_C.MODEL.TYPE = "tf_efficientnet_b4_ns"
_C.MODEL.ARGS = CN()
_C.MODEL.ARGS.num_classes = 5
_C.MODEL.ARGS.pretrained = True

_C.LOSS = "CrossEntropyLoss"

_C.METRICS = ["top_1_acc", "top_3_acc"]

_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = "Adam"
_C.OPTIMIZER.ARGS = CN()
_C.OPTIMIZER.ARGS.lr = 0.0001
_C.OPTIMIZER.ARGS.weight_decay = 0

_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.TYPE = "ReduceLROnPlateau"
_C.LR_SCHEDULER.ARGS = CN()
_C.LR_SCHEDULER.ARGS.factor = 0.2
_C.LR_SCHEDULER.ARGS.mode = "min"
_C.LR_SCHEDULER.ARGS.patience = 3
_C.LR_SCHEDULER.ARGS.verbose = True
_C.LR_SCHEDULER.ARGS.min_lr = 0.000001

_C.TRAINER = CN()
_C.TRAINER.epochs = 100
_C.TRAINER.save_period = 1