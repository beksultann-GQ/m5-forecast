"""Загрузка датасета M5 с Kaggle. Скриптом, не руками.

Требует KAGGLE_USERNAME / KAGGLE_KEY (см. .env.example) либо ~/.kaggle/kaggle.json.
Плюс на сайте нужно один раз принять правила соревнования — иначе API отдаст 403.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from m5.config import Config
from m5.utils.logging import get_logger
from m5.utils.paths import ensure_dir, resolve

logger = get_logger(__name__)

MANIFEST_NAME = "_MANIFEST.json"


class KaggleCredentialsError(RuntimeError):
    """Нет доступа к Kaggle API."""


def check_credentials() -> None:
    """Проверить, что креды есть. Падаем рано и с понятным текстом."""
    has_env = bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    if not (has_env or has_file):
        raise KaggleCredentialsError(
            "Нет Kaggle-кредов.\n"
            "  1. https://www.kaggle.com/settings -> API -> Create New Token\n"
            "  2. положи kaggle.json в ~/.kaggle/ (chmod 600)\n"
            "     либо впиши KAGGLE_USERNAME / KAGGLE_KEY в .env\n"
            "  3. на странице соревнования прими правила, иначе API вернёт 403"
        )


def download_competition(cfg: Config, force: bool = False) -> Path:
    """Скачать архив соревнования в data/raw и распаковать.

    Returns:
        Каталог с распакованными CSV.
    """
    raw_dir = ensure_dir(cfg.paths.raw_dir)
    expected = list(cfg.data.files.values())

    if not force and all((raw_dir / name).exists() for name in expected):
        logger.info("Все файлы уже в %s, скачивание пропускаю (--force чтобы перекачать)", raw_dir)
        write_manifest(raw_dir)
        return raw_dir

    check_credentials()
    competition = str(cfg.data.kaggle_competition)
    logger.info("Скачиваю %s в %s (~450 МБ)", competition, raw_dir)

    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition, "-p", str(raw_dir)],
            check=True,
        )
    except FileNotFoundError as exc:
        raise KaggleCredentialsError(
            "CLI `kaggle` не найден. Он ставится вместе с dev-зависимостями: `uv sync --all-extras`"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise KaggleCredentialsError(
            f"Kaggle вернул ошибку (код {exc.returncode}). Самая частая причина — "
            f"не приняты правила соревнования на https://www.kaggle.com/competitions/{competition}"
        ) from exc

    archive = raw_dir / f"{competition}.zip"
    if archive.exists():
        _unzip(archive, raw_dir)

    missing = [name for name in expected if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"После распаковки нет файлов: {missing}")

    write_manifest(raw_dir)
    return raw_dir


def _unzip(archive: Path, target_dir: Path) -> list[Path]:
    """Распаковать zip, вернуть список файлов."""
    ensure_dir(target_dir)
    logger.info("Распаковываю %s", archive.name)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target_dir)
        return [target_dir / name for name in zf.namelist()]


def verify_files(cfg: Config) -> dict[str, bool]:
    """Проверить наличие всех ожидаемых CSV. Используется сенсором в Airflow."""
    raw_dir = resolve(cfg.paths.raw_dir)
    return {name: (raw_dir / name).exists() for name in cfg.data.files.values()}


def write_manifest(raw_dir: Path) -> Path:
    """Манифест raw-слоя: имя, размер, sha256, mtime по каждому файлу.

    Это «версия данных» для воспроизводимости: тот же манифест = те же данные.
    Хеш логируем в MLflow вместе с прогоном — иначе через полгода не ответить,
    на каких именно данных обучалась прод-модель.
    """
    raw_dir = resolve(raw_dir)
    entries = []
    for path in sorted(raw_dir.glob("*.csv")):
        entries.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(),
            }
        )

    manifest = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "files": entries,
    }
    target = raw_dir / MANIFEST_NAME
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Манифест: %d файлов -> %s", len(entries), target.name)
    return target


def dataset_version(raw_dir: Path) -> str:
    """Короткий хеш манифеста — тег прогона в MLflow."""
    manifest = resolve(raw_dir) / MANIFEST_NAME
    if not manifest.exists():
        return "unknown"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    # created_at меняется при каждом перезапуске — в версию данных он не входит
    content = json.dumps(payload["files"], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Потоковый sha256: файлы до 500 МБ в память не тянем."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
