"""Логирование. Один setup на процесс, дальше get_logger(__name__)."""

from __future__ import annotations

import logging
import sys
from typing import Any

_CONFIGURED = False

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str | int = "INFO", fmt: str = DEFAULT_FORMAT) -> None:
    """Настроить корневой логгер. Идемпотентно."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Шумные библиотеки
    for noisy in ("httpx", "httpcore", "urllib3", "botocore", "mlflow.utils"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def log_step(logger: logging.Logger, step: str, **kwargs: Any) -> None:
    """Структурированная строка про шаг пайплайна — удобно грепать в Airflow."""
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info("step=%s %s", step, extra)
