"""Чтение из DuckDB: один слой доступа для baseline, обучения и инференса.

Вынесено из models/train.py специально: baseline'ы — чистый pandas и не должны
тянуть за собой LightGBM. Иначе `m5 model baseline` падает на машине без libomp,
хотя никакого бустинга там нет.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from m5.config import Config
from m5.data.duck import connect
from m5.utils.logging import get_logger

logger = get_logger(__name__)


def data_bounds(cfg: Config) -> tuple[date, date]:
    """Первая и последняя дата в staging-слое."""
    with connect(cfg.paths.duckdb_path, read_only=True) as con:
        row = con.execute("SELECT MIN(date), MAX(date) FROM stg.sales_enriched").fetchone()
    if not row or row[0] is None:
        raise RuntimeError("stg.sales_enriched пуста — сначала `make staging`")
    return row[0], row[1]


def load_sales(cfg: Config, start: date, end: date) -> pd.DataFrame:
    """Продажи (id, date, sales) за окно. Для baseline'ов и оценки качества."""
    with connect(cfg.paths.duckdb_path, read_only=True) as con:
        return con.execute(
            """
            SELECT id, date, CAST(sales AS DOUBLE) AS sales
            FROM stg.sales_enriched
            WHERE date BETWEEN ? AND ?
            """,
            [start, end],
        ).df()


def load_mart(
    cfg: Config,
    start: date,
    end: date,
    store_id: str | None = None,
) -> pd.DataFrame:
    """Витрина фич за окно дат, опционально по одному магазину.

    Грузим окном и по магазинам, а не целиком: на настоящем M5 витрина —
    десятки миллионов строк, одним куском в память она не поместится.
    """
    where = ["date BETWEEN ? AND ?"]
    params: list[Any] = [start, end]
    if store_id:
        where.append("store_id = ?")
        params.append(store_id)

    with connect(cfg.paths.duckdb_path, read_only=True) as con:
        return con.execute(f"SELECT * FROM ft.mart WHERE {' AND '.join(where)}", params).df()


def list_stores(cfg: Config) -> list[str]:
    """Магазины, присутствующие в витрине."""
    with connect(cfg.paths.duckdb_path, read_only=True) as con:
        rows = con.execute("SELECT DISTINCT store_id FROM ft.mart ORDER BY 1").fetchall()
    return [row[0] for row in rows]
