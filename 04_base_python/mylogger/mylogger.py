import datetime
import logging
from pathlib import Path


def create_logger():
    logger_name = f"{datetime.datetime.now().strftime('%Y-%m-%d')}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger_formatter = '%(asctime)s - %(filename)s:%(lineno)d - [%(levelname)s]:%(message)s'

    # create console output
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(logger_formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # create console output filehandler
    log_dir = Path(__file__).parent.parent / 'mylogger' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{logger_name}.log"
    file_handler = logging.FileHandler(str(log_file))
    formatter = logging.Formatter(logger_formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


mylogger = create_logger()
