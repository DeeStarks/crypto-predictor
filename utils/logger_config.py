import logging
import sys
from pathlib import Path


def setup_logging(log_level="INFO", log_file=None, quiet=False):
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))

    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
