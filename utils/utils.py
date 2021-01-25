import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
import numpy as np
import os
import random
import yaml

LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
RUN_DIR = "runs"


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_config(fname):
    fname = Path(fname)
    with fname.open('rt') as fh:
        return yaml.safe_load(fh)


def write_config(content, fname):
    fname = Path(fname)
    with fname.open('wt') as fh:
        yaml.dump(content, fh, indent=4, sort_keys=False)


def ensure_exists(p: Path) -> Path:
    """
    Helper to ensure a directory exists.
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def arch_path(config: dict) -> Path:
    """
    Construct a path based on the name of a configuration file eg. 'saved/EfficientNet'
    """
    p = Path(config["save_dir"]) / config["name"]
    return ensure_exists(p)


def log_path(config: dict) -> Path:
    p = arch_path(config) / LOG_DIR
    return ensure_exists(p)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.scatter(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['Ameme'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False