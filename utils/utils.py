import random
import os
import numpy as np
import torch


def get_instance(module, node_name, node_params, *args):
    return getattr(module, node_name)(*args, **node_params)
