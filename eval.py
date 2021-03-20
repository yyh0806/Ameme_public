import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import loss.loss as module_loss
import model.metric as module_metric
from utils import prepare_device, seed_everything, MyEnsemble
from utils import MetricTracker
from tqdm import tqdm
import logging


class Eval:

    def __init__(self, models, criterion, metrics, device):
        self.criterion = criterion
        self.models = models
        self.device = device
        self.metrics = metrics
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metrics], writer=None)
        self.logger = logging.getLogger()

    def eval(self, valid_data_loader):
        for model in self.models:
            model.eval()
        self.valid_metrics.reset()
        outputs = []
        targets = []
        with torch.no_grad():
            tk = tqdm(enumerate(valid_data_loader), total=len(valid_data_loader))
            for batch_idx, (data, target) in tk:
                data, target = data.to(self.device), target.to(self.device)
                for model in self.models:
                    output = model(data)
                    output2 = model(data.flip(-1))
                    loss = self.criterion(output, target)

                    outputs.append(
                        (output.sigmoid().detach().cpu().numpy() + output2.sigmoid().detach().cpu().numpy()) / 2)
                    targets.append(target.cpu().numpy())

                self.valid_metrics.update('loss', loss.item())
                tk.set_description("loss: %.6f" % loss.item())
        outputs = np.concatenate(outputs)
        targets = np.concatenate(targets)
        for met in self.metrics:
            self.valid_metrics.update(met.__name__, met(outputs, targets))
        self.logger.info(self.valid_metrics.result())
        return self.valid_metrics.result()
