import os
import yaml
import logging.config

logging_level_dict = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}


def setup_logging(default_level=logging.INFO):
    logging.basicConfig(level=default_level)


def setup_logger(cls, verbose=0):
    logger = logging.getLogger(cls.__class__.__name__)
    if verbose not in logging_level_dict:
        raise KeyError(f'verbose option {verbose} for {cls} not valid. '
                       f'Valid options are {logging_level_dict.keys()}.')
    logger.setLevel(logging_level_dict[verbose])
    return logger

