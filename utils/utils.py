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
from torch.nn.modules.loss import _Loss


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


def arch_path(string: str) -> Path:
    """
    Construct a path based on the name of a configuration file eg. 'saved/EfficientNet'
    """
    return ensure_exists(string)


def log_path(string: str) -> Path:
    return ensure_exists(string)


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


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1., use_cuda=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    size = x.size()
    bbx1, bby1, bbx2, bby2 = rand_bbox(size, lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


class BiTemperedLogisticLoss(_Loss):
    def __init__(self, reduction='mean', t1=1, t2=1, label_smoothing=0.0, num_iters=5):
        super().__init__(reduction=reduction)
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.num_iters = num_iters

    @classmethod
    def log_t(cls, u, t):
        """Compute log_t for `u`."""

        if t == 1.0:
            return torch.log(u)
        else:
            return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    @classmethod
    def exp_t(cls, u, t):
        """Compute exp_t for `u`."""

        if t == 1.0:
            return torch.exp(u)
        else:
            return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    @classmethod
    def compute_normalization_fixed_point(cls, activations, t, num_iters=5):
        """Returns the normalization value for each example (t > 1.0).
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (> 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        mu = torch.max(activations, dim=-1).values.view(-1, 1)
        normalized_activations_step_0 = activations - mu

        normalized_activations = normalized_activations_step_0
        i = 0
        while i < num_iters:
            i += 1
            logt_partition = torch.sum(cls.exp_t(normalized_activations, t), dim=-1).view(-1, 1)
            normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

        logt_partition = torch.sum(cls.exp_t(normalized_activations, t), dim=-1).view(-1, 1)

        return -cls.log_t(1.0 / logt_partition, t) + mu

    @classmethod
    def compute_normalization(cls, activations, t, num_iters=5):
        """Returns the normalization value for each example.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
        num_iters: Number of iterations to run the method.
        Return: A tensor of same rank as activation with the last dimension being 1.
        """

        if t < 1.0:
            return None  # not implemented as these values do not occur in the authors experiments...
        else:
            return cls.compute_normalization_fixed_point(activations, t, num_iters)

    @classmethod
    def tempered_softmax(cls, activations, t, num_iters=5):
        """Tempered softmax function.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        t: Temperature tensor > 0.0.
        num_iters: Number of iterations to run the method.
        Returns:
        A probabilities tensor.
        """
        if t == 1.0:
            normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
        else:
            normalization_constants = cls.compute_normalization(activations, t, num_iters)

        return cls.exp_t(activations - normalization_constants, t)

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        """Bi-Tempered Logistic Loss with custom gradient.
        Args:
        activations: A multi-dimensional tensor with last dimension `num_classes`.
        labels: A tensor with shape and dtype as activations.
        t1: Temperature 1 (< 1.0 for boundedness).
        t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
        label_smoothing: Label smoothing parameter between [0, 1).
        num_iters: Number of iterations to run the method.
        Returns:
        A loss tensor.
        """
        if self.label_smoothing > 0.0:
            targets = BiTemperedLogisticLoss._smooth_one_hot(targets, inputs.size(-1), self.label_smoothing)

        probabilities = self.tempered_softmax(inputs, self.t2, self.num_iters)

        temp1 = (self.log_t(targets + 1e-10, self.t1) - self.log_t(probabilities, self.t1)) * targets
        temp2 = (1 / (2 - self.t1)) * (torch.pow(targets, 2 - self.t1) - torch.pow(probabilities, 2 - self.t1))
        loss = temp1 - temp2

        loss = loss.sum(dim=-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss