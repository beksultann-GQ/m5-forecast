"""Единая точка правды про пути. Никаких относительных путей по коду."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Корень репозитория.

    Ищем вверх от этого файла до каталога с pyproject.toml.
    Переопределяется переменной окружения M5_REPO_ROOT (нужно в Airflow-контейнере).
    """
    env = os.getenv("M5_REPO_ROOT")
    if env:
        return Path(env).resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return here.parents[3]


def resolve(path: str | Path) -> Path:
    """Абсолютный путь: относительные считаются от корня репо."""
    p = Path(path)
    return p if p.is_absolute() else (repo_root() / p)


def ensure_dir(path: str | Path) -> Path:
    """Создать каталог (и родителей), вернуть абсолютный путь."""
    p = resolve(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: str | Path) -> Path:
    """Создать каталог-родитель для файла, вернуть абсолютный путь к файлу."""
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
