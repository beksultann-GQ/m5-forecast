"""Обёртка над DuckDB: подключение, выполнение .sql-файлов, регистрация parquet.

Вся тяжёлая работа (unpivot 59 млн строк, join'ы, оконные функции) живёт здесь,
Python — только клей.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

from m5.utils.logging import get_logger
from m5.utils.paths import ensure_parent, resolve

logger = get_logger(__name__)

# Плейсхолдеры вида {{ raw_dir }} в .sql-файлах
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@contextmanager
def connect(
    db_path: str | Path | None = None,
    read_only: bool = False,
    memory_limit: str = "6GB",
    threads: int | None = None,
    temp_dir: str | Path | None = None,
) -> Iterator[duckdb.DuckDBPyConnection]:
    """Подключение к DuckDB с настройками под ноутбук.

    memory_limit + temp_directory дают out-of-core: то, что не влезло в RAM,
    спиллится на диск, а не убивает процесс.
    """
    target = str(ensure_parent(db_path)) if db_path else ":memory:"
    con = duckdb.connect(target, read_only=read_only)
    try:
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute("SET preserve_insertion_order=false")
        if threads:
            con.execute(f"SET threads={threads}")
        if temp_dir:
            con.execute(f"SET temp_directory='{resolve(temp_dir)}'")
        yield con
    finally:
        con.close()


def render_sql(sql: str, params: Mapping[str, Any]) -> str:
    """Подставить {{ placeholders }}. Только для путей/имён схем, не для данных."""

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in params:
            raise KeyError(f"В SQL есть {{{{ {key} }}}}, но его нет в params")
        return str(params[key])

    return _PLACEHOLDER.sub(_sub, sql)


def run_sql_file(
    con: duckdb.DuckDBPyConnection,
    path: str | Path,
    params: Mapping[str, Any] | None = None,
) -> None:
    """Выполнить .sql-файл (может содержать несколько стейтментов)."""
    sql_path = resolve(path)
    sql = sql_path.read_text(encoding="utf-8")
    if params:
        sql = render_sql(sql, params)

    logger.info("Выполняю SQL: %s", sql_path.name)
    con.execute(sql)


def run_sql_dir(
    con: duckdb.DuckDBPyConnection,
    directory: str | Path,
    params: Mapping[str, Any] | None = None,
    pattern: str = "*.sql",
) -> list[Path]:
    """Выполнить все .sql из каталога в лексикографическом порядке.

    Поэтому файлы нумеруем: 00_, 01_, 10_ — порядок исполнения виден в `ls`.
    """
    files = sorted(resolve(directory).glob(pattern))
    for f in files:
        run_sql_file(con, f, params)
    return files


def create_schemas(con: duckdb.DuckDBPyConnection, schemas: list[str]) -> None:
    """Слои как в DWH: raw / stg / ft / mart."""
    for schema in schemas:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")


def export_parquet(
    con: duckdb.DuckDBPyConnection,
    query: str,
    out_path: str | Path,
    partition_by: list[str] | None = None,
    compression: str = "zstd",
) -> Path:
    """Выгрузить результат запроса в parquet (опционально с партиционированием)."""
    target = ensure_parent(out_path)
    opts = ["FORMAT PARQUET", f"COMPRESSION {compression}"]
    if partition_by:
        opts.append(f"PARTITION_BY ({', '.join(partition_by)})")
        opts.append("OVERWRITE_OR_IGNORE")
    con.execute(f"COPY ({query}) TO '{target}' ({', '.join(opts)})")
    logger.info("Выгружено в %s", target)
    return target


def table_stats(con: duckdb.DuckDBPyConnection, table: str) -> dict[str, Any]:
    """Быстрая сводка по таблице — печатаем в логи после каждого шага."""
    row = con.execute(f"SELECT COUNT(*) AS n_rows FROM {table}").fetchone()
    cols = con.execute(f"DESCRIBE {table}").fetchall()
    return {"table": table, "n_rows": row[0] if row else 0, "n_cols": len(cols)}
