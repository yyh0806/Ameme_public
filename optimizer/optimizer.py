from torch import optim as op


# Adam: A Method for Stochastic Optimization(https://arxiv.org/abs/1412.6980)
def Adam(lr: float = 0.0002, betas: tuple = (0.9, 0.999), eps: float = 1e-08, weight_decay: float = 0, **kwargs):
    optim = op.Adam(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs)
    return optim
