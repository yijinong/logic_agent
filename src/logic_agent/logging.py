import logging

from rich.logging import RichHandler

LOGGERS = {}


class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelname == "WARNING":
            record.levelname = "WARN"
        return super().format(record)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    if name in LOGGERS:
        return LOGGERS[name]

    fmt = "%(message)s"
    formatter = CustomFormatter(fmt=fmt)
    handler = RichHandler(level=level)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)

    # fmt = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    # datefmt = "%Y-%m-%d %H:%M:%S"
    # formatter = CustomFormatter(fmt=fmt, datefmt=datefmt)
    # handler = logging.StreamHandler()
    # handler.setFormatter(formatter)
    # handler.setLevel(level)

    # logger = logging.getLogger(name)
    # logger.propagate = False
    # logger.setLevel(level)
    # logger.handlers.clear()
    # logger.addHandler(handler)

    LOGGERS[name] = logger
    return logger
