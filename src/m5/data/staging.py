"""Staging: wide -> long + справочники. Всё на DuckDB SQL.

sales_train_evaluation лежит в широком формате (d_1 ... d_1941).
После разворота — ~59 млн строк (30490 рядов × 1941 день). pd.melt такое не тянет,
DuckDB UNPIVOT — тянет, потому что работает out-of-core.

Python здесь только оркестрирует: считает параметры, вызывает .sql-файлы,
печатает статистику. Логика — в src/m5/data/sql/.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from m5.config import Config
from m5.data.duck import connect, run_sql_file, table_stats
from m5.utils.logging import get_logger
from m5.utils.paths import resolve

logger = get_logger(__name__)

SQL_DIR = Path(__file__).parent / "sql"

SQL_FILES = [
    "00_stg_calendar.sql",
    "01_stg_sales_long.sql",
    "02_stg_prices.sql",
    "03_stg_sales_enriched.sql",
]

STAGING_TABLES = [
    "stg.calendar",
    "stg.snap",
    "stg.calendar_events",
    "stg.hierarchy",
    "stg.sales_long",
    "stg.first_sale",
    "stg.prices",
    "stg.prices_relative",
    "stg.sales_enriched",
]


def build_staging(cfg: Config) -> dict[str, int]:
    """Прогнать staging-слой: raw parquet -> схема stg в DuckDB.

    Returns:
        {имя_таблицы: число_строк}
    """
    raw_dir = resolve(cfg.paths.raw_dir)
    params = {
        "raw_dir": str(raw_dir),
        "history_start_date": history_start_date(cfg).isoformat(),
    }
    logger.info("Staging с параметрами: %s", params)

    stats: dict[str, int] = {}
    with connect(cfg.paths.duckdb_path, temp_dir=cfg.paths.interim_dir) as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS stg")
        for filename in SQL_FILES:
            run_sql_file(con, SQL_DIR / filename, params)

        for table in STAGING_TABLES:
            info = table_stats(con, table)
            stats[table] = info["n_rows"]
            logger.info("%s: %s строк × %d колонок", table, f"{info['n_rows']:,}", info["n_cols"])

    return stats


def history_start_date(cfg: Config) -> date:
    """С какой даты держим историю в витрине.

    cfg.data.keep_last_days = null -> вся история с origin_date.
    Иначе обрезаем: старше двух-трёх лет данные почти не влияют на прогноз,
    а память и время сборки фич едят линейно.
    """
    origin = date.fromisoformat(str(cfg.data.origin_date))
    keep = cfg.data.get("keep_last_days")
    if not keep:
        return origin

    last = day_index_to_date(cfg, int(cfg.data.last_day_index))
    start = last - timedelta(days=int(keep))
    return max(origin, start)


def day_index_to_date(cfg: Config, day_index: int) -> date:
    """'d_1914' -> дата. d_1 соответствует cfg.data.origin_date."""
    origin = date.fromisoformat(str(cfg.data.origin_date))
    return origin + timedelta(days=day_index - 1)


def date_to_day_index(cfg: Config, target: date) -> int:
    """Обратное преобразование."""
    origin = date.fromisoformat(str(cfg.data.origin_date))
    return (target - origin).days + 1


def max_date(cfg: Config) -> date:
    """Последняя дата с фактическими продажами. Дефолт для as_of."""
    with connect(cfg.paths.duckdb_path, read_only=True) as con:
        row = con.execute("SELECT MAX(date) FROM stg.sales_enriched").fetchone()
    if not row or row[0] is None:
        raise RuntimeError("stg.sales_enriched пуста — сначала `make staging`")
    return row[0]


def export_staging_parquet(cfg: Config, out_dir: Path | None = None) -> Path:
    """Выгрузить stg.sales_enriched в партиционированный по магазинам parquet.

    Обучение идёт по магазинам, так каждая модель читает только свой кусок.
    """
    from m5.data.duck import export_parquet

    target = resolve(out_dir or Path(cfg.paths.interim_dir) / "sales_enriched")
    with connect(cfg.paths.duckdb_path, read_only=True) as con:
        return export_parquet(
            con,
            "SELECT * FROM stg.sales_enriched",
            target,
            partition_by=["store_id"],
        )
