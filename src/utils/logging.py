import logging
import re
import sys
import warnings
from typing import List


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    handlers = get_handlers()
    logger.handlers = handlers
    logger.setLevel(level)

    return logger


def is_logging_disabled(logger: logging.Logger):
    return not logger.isEnabledFor(logging.INFO)


def get_handlers() -> List[logging.StreamHandler]:
    var = "%(asctime)s %(levelname)s %(name)s - %(message)s"
    formatter = logging.Formatter(var)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.addFilter(lambda record: record.levelno > logging.INFO)

    return [stdout_handler, stderr_handler]


def _configure_warnings_logger():
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    def formatwarning(
        msg: UserWarning, category: type, filename: str, lineno: int, *args
    ) -> str:
        msg = re.sub(r"\s+", " ", str(msg))
        return f"{filename}:{lineno} - {msg}"

    warnings.formatwarning = formatwarning
    logging.captureWarnings(True)

    logger = logging.getLogger("py.warnings")
    logger.handlers = [stderr_handler]
    logger.propagate = False


_configure_warnings_logger()
