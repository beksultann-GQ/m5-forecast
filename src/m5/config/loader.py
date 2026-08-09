"""Загрузка yaml-конфигов. Ничего не хардкодим в коде.

Использование:
    cfg = load_config()                       # conf/config.yaml + всё из defaults
    cfg.model.params.learning_rate            # точечный доступ
    cfg = load_config(overrides=["model.params.learning_rate=0.01"])
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from m5.utils.paths import repo_root, resolve

Config = DictConfig

DEFAULT_CONF_DIR = "conf"
ROOT_FILE = "config.yaml"


def load_config(
    conf_dir: str | Path = DEFAULT_CONF_DIR,
    root_file: str = ROOT_FILE,
    overrides: Sequence[str] | None = None,
) -> Config:
    """Собрать конфиг из conf/.

    Порядок: файлы из `defaults` (в порядке списка) -> корневой config.yaml -> overrides.
    Корневой файл кладётся поверх, чтобы в нём можно было точечно переопределять.
    """
    conf_path = resolve(conf_dir)
    root_cfg = OmegaConf.load(conf_path / root_file)

    parts: list[DictConfig] = []
    for name in root_cfg.pop("defaults", []) or []:
        f = conf_path / f"{name}.yaml"
        if not f.exists():
            raise FileNotFoundError(f"В defaults указан {name}, но нет файла {f}")
        parts.append(OmegaConf.load(f))

    cfg = OmegaConf.merge(*parts, root_cfg) if parts else root_cfg

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    _absolutize_paths(cfg)
    OmegaConf.resolve(cfg)
    return cfg  # type: ignore[return-value]


def _absolutize_paths(cfg: DictConfig) -> None:
    """Все пути в cfg.paths делаем абсолютными относительно корня репо."""
    if "paths" not in cfg:
        return
    for key, value in cfg.paths.items():
        if isinstance(value, str):
            cfg.paths[key] = str(resolve(value))


def to_dict(cfg: Config) -> dict[str, Any]:
    """Обычный dict — для логирования в MLflow."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def flatten(cfg: Config, prefix: str = "") -> dict[str, Any]:
    """Плоский dict `a.b.c -> value`. MLflow.log_params не умеет вложенность."""
    flat: dict[str, Any] = {}
    for key, value in to_dict(cfg).items():
        _flatten_into(flat, f"{prefix}{key}", value)
    return flat


def _flatten_into(acc: dict[str, Any], key: str, value: Any) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_into(acc, f"{key}.{k}", v)
    elif isinstance(value, list) and value and isinstance(value[0], dict):
        acc[key] = str(value)
    else:
        acc[key] = value


def config_hash(cfg: Config) -> str:
    """Стабильный хеш конфига — тегом в MLflow, чтобы отличать прогоны."""
    import hashlib
    import json

    payload = json.dumps(to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def git_sha() -> str:
    """SHA текущего коммита. Логируем вместе с прогоном."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
