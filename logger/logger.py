import yaml
import logging
import logging.config
from pathlib import Path
from utils import log_path


def setup_logging(run_config, log_config="logger/logger_config.yml", LOG_LEVEL=logging.INFO) -> None:
    """
    Setup ``logging.config``

    Parameters
    ----------
    run_config : str
        Path to configuration file for run

    log_config : str
        Path to configuration file for logging
        :param run_config:
        :param log_config:
        :param LOG_LEVEL:
    """
    log_config = Path(log_config)

    if not log_config.exists():
        logging.basicConfig(level=LOG_LEVEL)
        logger = logging.getLogger("setup")
        logger.warning(f'"{log_config}" not found. Using basicConfig.')
        return

    with open(log_config, "rt") as f:
        config = yaml.safe_load(f.read())

    # modify logging paths based on run config
    run_path = log_path(run_config)
    for _, handler in config["handlers"].items():
        if "filename" in handler:
            handler["filename"] = str(run_path / handler["filename"])

    logging.config.dictConfig(config)
