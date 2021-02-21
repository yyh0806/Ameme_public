from torch import optim as op


# Adam: A Method for Stochastic Optimization(https://arxiv.org/abs/1412.6980)
def Adam(lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, **kwargs):
    optim = op.Adam(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **kwargs)
    return optim
