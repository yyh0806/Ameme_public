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
_C.MODEL.ARGS.drop_rate = 0.0
_C.MODEL.ARGS.drop_connect_rate = None  # DEPRECATED, use drop_path
_C.MODEL.ARGS.drop_path_rate = None
_C.MODEL.ARGS.drop_block_rate = None
_C.MODEL.ARGS.global_pool = None  # (fast, avg, max, avgmax, avgmaxc)
_C.MODEL.ARGS.bn_tf = None  # efficientnet, mobilenetv3
_C.MODEL.ARGS.bn_momentum = None  # efficientnet, mobilenetv3
_C.MODEL.ARGS.bn_eps = None  # efficientnet, mobilenetv3
_C.MODEL.ARGS.scriptable = False  # efficientnet, mobilenetv3
_C.MODEL.ARGS.checkpoint_path = ""

_C.LOSS = "CrossEntropyLoss"

_C.METRICS = ["top_1_acc", "top_3_acc"]

_C.OPTIMIZER = CN()
_C.OPTIMIZER.TYPE = "Adam"
_C.OPTIMIZER.ARGS = CN()
_C.OPTIMIZER.ARGS.lr = 0.0001
_C.OPTIMIZER.ARGS.weight_decay = 0.0001
_C.OPTIMIZER.ARGS.opt_eps = None
_C.OPTIMIZER.ARGS.opt_betas = None
_C.OPTIMIZER.ARGS.momentum = 0.9
_C.OPTIMIZER.ARGS.clip_grad = None
_C.OPTIMIZER.ARGS.clip_mode = "norm"
_C.OPTIMIZER.ARGS.filter_bias_and_bn = True


_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.TYPE = "cosine"  # (cosine, tanh, step, plateau)
_C.LR_SCHEDULER.ARGS = CN()
_C.LR_SCHEDULER.ARGS.epochs = 200
_C.LR_SCHEDULER.ARGS.start_epoch = None
_C.LR_SCHEDULER.ARGS.min_lr = 1e-5
_C.LR_SCHEDULER.ARGS.decay_epochs = 30
_C.LR_SCHEDULER.ARGS.decay_rate = 0.1
_C.LR_SCHEDULER.ARGS.warmup_lr = 0.0001
_C.LR_SCHEDULER.ARGS.warmup_epochs = 3
_C.LR_SCHEDULER.ARGS.cooldown_epochs = 10
_C.LR_SCHEDULER.ARGS.patience_epochs = 10
_C.LR_SCHEDULER.ARGS.lr_noise = None
_C.LR_SCHEDULER.ARGS.lr_noise_pct = 0.67
_C.LR_SCHEDULER.ARGS.lr_noise_std = 1.0
_C.LR_SCHEDULER.ARGS.lr_cycle_mul = 1.0
_C.LR_SCHEDULER.ARGS.lr_cycle_limit = 1
_C.LR_SCHEDULER.ARGS.seed = 86


_C.TRAINER = CN()
_C.TRAINER.epochs = 100
_C.TRAINER.save_period = 1
