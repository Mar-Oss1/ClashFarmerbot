"""
core/logger.py
Logger centralizado con salida formateada a consola y archivo.
"""
import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from core.config import PROJECT_ROOT

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "clash_bot.log")

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-16s | %(message)s"
_DATE_FMT = "%H:%M:%S"
_ROOT_LOGGER_NAME = "clashbot"
_lock = threading.Lock()


def _ensure_root_logger() -> logging.Logger:
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    if root.handlers:
        return root
    with _lock:
        if root.handlers:
            return root
        root.setLevel(logging.INFO)
        formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FMT)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        root.propagate = False
        return root


def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    _ensure_root_logger()
    if not name:
        full_name = _ROOT_LOGGER_NAME
    elif name.startswith(f"{_ROOT_LOGGER_NAME}.") or name == _ROOT_LOGGER_NAME:
        full_name = name
    else:
        full_name = f"{_ROOT_LOGGER_NAME}.{name}"

    logger = logging.getLogger(full_name)
    logger.setLevel(level)
    logger.propagate = True
    return logger
