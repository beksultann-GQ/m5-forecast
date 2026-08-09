"""CSV -> Parquet. Raw-слой immutable: сконвертировали один раз и не трогаем.

Зачем: sales_train_evaluation.csv ~120 МБ и 1947 колонок, читать его CSV-парсером
на каждом шаге — потеря минут. Parquet + zstd читается в разы быстрее и жмётся в ~10 раз.

Конвертируем через DuckDB, а не pandas: pandas материализует весь файл в RAM,
DuckDB стримит из CSV прямо в parquet.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from m5.config import Config
from m5.data.duck import connect
from m5.utils.logging import get_logger
from m5.utils.paths import ensure_dir, resolve

logger = get_logger(__name__)

# Явные типы вместо автодетекта там, где мы знаем полный список колонок.
# sales_train_evaluation не описываем: 1947 колонок, автодетект справляется.
DTYPE_HINTS: dict[str, dict[str, str]] = {
    "calendar": {
        "date": "DATE",
        "wm_yr_wk": "INTEGER",
        "weekday": "VARCHAR",
        "wday": "TINYINT",
        "month": "TINYINT",
        "year": "SMALLINT",
        "d": "VARCHAR",
        "event_name_1": "VARCHAR",
        "event_type_1": "VARCHAR",
        "event_name_2": "VARCHAR",
        "event_type_2": "VARCHAR",
        "snap_CA": "TINYINT",
        "snap_TX": "TINYINT",
        "snap_WI": "TINYINT",
    },
    "sell_prices": {
        "store_id": "VARCHAR",
        "item_id": "VARCHAR",
        "wm_yr_wk": "INTEGER",
        "sell_price": "FLOAT",
    },
}


def csv_to_parquet(cfg: Config, overwrite: bool = False) -> dict[str, Path]:
    """Сконвертировать все CSV из cfg.data.files в parquet.

    Returns:
        {логическое_имя: путь_к_parquet}
    """
    raw_dir = ensure_dir(cfg.paths.raw_dir)
    result: dict[str, Path] = {}

    with connect() as con:
        for name, filename in cfg.data.files.items():
            csv_path = raw_dir / filename
            parquet_path = parquet_path_for(cfg, name)

            if not csv_path.exists():
                raise FileNotFoundError(
                    f"Нет файла {csv_path}. Сначала `make download` (или `m5 data synth`)"
                )

            if parquet_path.exists() and not overwrite and _is_fresh(csv_path, parquet_path):
                logger.info("%s: parquet свежее csv, пропускаю", name)
                result[name] = parquet_path
                continue

            convert_one(csv_path, parquet_path, DTYPE_HINTS.get(name), con=con)
            stats = con.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_path}')").fetchone()
            n_cols = len(
                con.execute(f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')").fetchall()
            )
            logger.info(
                "%s: %s строк × %d колонок -> %s", name, f"{stats[0]:,}", n_cols, parquet_path.name
            )
            result[name] = parquet_path

    return result


def convert_one(
    csv_path: Path,
    parquet_path: Path,
    dtypes: dict[str, str] | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
) -> Path:
    """Сконвертировать один CSV. Вынесено отдельно ради тестов на мини-файлах."""
    csv_path = resolve(csv_path)
    parquet_path = resolve(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    if dtypes:
        columns = ", ".join(f"'{col}': '{dtype}'" for col, dtype in dtypes.items())
        reader = f"read_csv('{csv_path}', header=true, columns={{{columns}}})"
    else:
        # sample_size=-1: сканируем весь файл при определении типов.
        # На широкой таблице выборки в 20 тыс. строк не хватает, и колонка,
        # где нули идут первые 20 тыс. строк, определится как BOOLEAN.
        reader = f"read_csv_auto('{csv_path}', header=true, sample_size=-1)"

    owns_connection = con is None
    con = con or duckdb.connect(":memory:")
    try:
        con.execute(
            f"COPY (SELECT * FROM {reader}) TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION zstd)"
        )
    finally:
        if owns_connection:
            con.close()

    return parquet_path


def parquet_path_for(cfg: Config, logical_name: str) -> Path:
    """Путь к parquet по логическому имени ('calendar' -> data/raw/calendar.parquet)."""
    return Path(cfg.paths.raw_dir) / f"{logical_name}.parquet"


def _is_fresh(csv_path: Path, parquet_path: Path) -> bool:
    """Parquet новее исходного CSV — конвертировать заново незачем."""
    return parquet_path.stat().st_mtime >= csv_path.stat().st_mtime
