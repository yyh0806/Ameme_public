from optimizer.optimizers import *
from torch import optim as optim


def Adam(parameters, opt_args):
    optimizer = optim.Adam(parameters, **opt_args)
    return optimizer
