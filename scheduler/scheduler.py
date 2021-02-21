from torch.optim.lr_scheduler import ReduceLROnPlateau


def ReduceLR(mode='min', factor=0.1, patience=10,
                      threshold=0.0001, threshold_mode='rel',
                      cooldown=0, min_lr=0, eps=1e-08, verbose=False, **kwargs):
    scheduler = ReduceLROnPlateau(mode=mode, factor=factor, patience=patience,
                                  threshold=threshold, threshold_mode=threshold_mode,
                                  cooldown=cooldown, min_lr=min_lr, eps=eps, verbose=verbose, **kwargs)
    return scheduler
