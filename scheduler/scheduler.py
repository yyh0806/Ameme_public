from torch.optim.lr_scheduler import ReduceLROnPlateau


def ReduceLR(mode: str = 'min', factor: float = 0.1, patience: int = 10,
             threshold: float = 0.0001, threshold_mode: str = 'rel',
             cooldown: int = 0, min_lr: float = 0, eps: float = 1e-08, verbose: bool = False, **kwargs):
    scheduler = ReduceLROnPlateau(mode=mode, factor=factor, patience=patience,
                                  threshold=threshold, threshold_mode=threshold_mode,
                                  cooldown=cooldown, min_lr=min_lr, eps=eps, verbose=verbose, **kwargs)
    return scheduler
