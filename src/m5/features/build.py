"""Сборка витрины фич. Оркестрация SQL-файлов, вся логика — в них.

КЛЮЧЕВОЕ ТРЕБОВАНИЕ PRODUCTION-READY:
    Одна и та же функция считает фичи и для train, и для inference.
    Не два скрипта. Иначе получится training/serving skew — классический прод-баг,
    когда модель на валидации прекрасна, а в проде даёт мусор, потому что
    в inference-скрипте кто-то посчитал rolling mean чуть иначе.

    Здесь это обеспечено так: build_features(as_of=...) — единственная точка входа.
    Разница между train и inference только в параметре as_of (граница «что мы знаем»)
    и в наличии таргета. SQL — один и тот же файл.
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

FEATURE_TABLES = ["ft.lags", "ft.rolling", "ft.rolling_agg", "ft.weather", "ft.holidays", "ft.mart"]


class LeakageError(RuntimeError):
    """В фичах обнаружены данные из будущего."""


def build_features(
    cfg: Config,
    as_of: date | str | None = None,
    mode: str = "train",
    stores: list[str] | None = None,
) -> dict[str, int]:
    """Собрать витрину фич по состоянию на дату `as_of`.

    Args:
        cfg: конфиг
        as_of: граница знания. Все фичи считаются ТОЛЬКО по данным <= as_of.
            None -> максимальная дата в stg.sales_enriched.
        mode: 'train' | 'inference'
        stores: собирать только по этим магазинам (экономия памяти и времени)

    Returns:
        {имя_таблицы: число_строк}

    Гарантии, которые держит эта функция:
        1. Ни одна фича не смотрит в данные позже as_of.
        2. Все лаги/окна сдвинуты минимум на horizon дней.
        3. Один и тот же SQL для обоих режимов.
    """
    from m5.data.staging import max_date

    if mode not in ("train", "inference"):
        raise ValueError(f"mode должен быть 'train' или 'inference', получено {mode!r}")

    as_of = _resolve_as_of(cfg, as_of, max_date)
    horizon = int(cfg.project.horizon)

    # Единственное, чем отличаются train и inference: до какой даты тянем витрину.
    # На инференсе добавляем горизонт — это те самые будущие строки с sales = NULL,
    # по которым будет предсказание. SQL при этом один и тот же.
    ft_max_date = as_of if mode == "train" else as_of + timedelta(days=horizon)

    params = {
        "as_of": as_of.isoformat(),
        "ft_max_date": ft_max_date.isoformat(),
        "horizon": horizon,
        "external_dir": str(resolve(cfg.paths.external_dir)),
        "store_filter": _store_filter(stores),
    }
    logger.info(
        "Сборка фич: mode=%s as_of=%s ft_max_date=%s stores=%s",
        mode,
        as_of,
        ft_max_date,
        stores or "все",
    )

    sql_files = _sql_plan(cfg)
    stats: dict[str, int] = {}

    with connect(cfg.paths.duckdb_path, temp_dir=cfg.paths.interim_dir) as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS ft")
        for filename in sql_files:
            run_sql_file(con, SQL_DIR / filename, params)

        for table in FEATURE_TABLES:
            info = table_stats(con, table)
            stats[table] = info["n_rows"]
            logger.info("%s: %s строк × %d колонок", table, f"{info['n_rows']:,}", info["n_cols"])

        assert_no_leakage(con, as_of)

    return stats


def _sql_plan(cfg: Config) -> list[str]:
    """Какие .sql выполнять. Внешние источники подменяются заглушками, если их нет.

    Состав колонок витрины от этого не меняется — см. комментарий в *_stub.sql.
    """
    external_dir = resolve(cfg.paths.external_dir)
    weather_ready = (external_dir / "weather.parquet").exists()
    holidays_ready = (external_dir / "holidays.parquet").exists()

    if not weather_ready:
        logger.warning("Нет %s — погодные фичи будут NULL", external_dir / "weather.parquet")
    if not holidays_ready:
        logger.warning("Нет %s — праздничные фичи будут NULL", external_dir / "holidays.parquet")

    return [
        "00_ft_lags.sql",
        "01_ft_rolling.sql",
        "02_ft_weather.sql" if weather_ready else "02_ft_weather_stub.sql",
        "03_ft_holidays.sql" if holidays_ready else "03_ft_holidays_stub.sql",
        "10_ft_mart.sql",
    ]


def _store_filter(stores: list[str] | None) -> str:
    """SQL-условие фильтра по магазинам. Подставляется в WHERE витрины."""
    if not stores:
        return "TRUE"
    quoted = ", ".join(f"'{store}'" for store in stores)
    return f"e.store_id IN ({quoted})"


def _resolve_as_of(cfg: Config, as_of: date | str | None, fallback) -> date:
    """as_of как дата: строка из CLI, дата из кода или максимум из данных."""
    if as_of is None:
        return fallback(cfg)
    if isinstance(as_of, str):
        return date.fromisoformat(as_of)
    return as_of


def feature_columns(cfg: Config, con=None) -> list[str]:
    """Список колонок-фич, которые уходят в модель.

    Единый источник правды: и train, и predict берут порядок колонок отсюда.
    Если порядок разъедется — LightGBM молча предскажет ерунду.

    Берём фактический состав витрины минус служебные колонки: так список
    не может разойтись с тем, что реально лежит в ft.mart.
    """
    owns_connection = con is None
    con = con or connect(cfg.paths.duckdb_path, read_only=True).__enter__()
    try:
        columns = [row[0] for row in con.execute("DESCRIBE ft.mart").fetchall()]
    finally:
        if owns_connection:
            con.close()

    return [col for col in columns if col not in NON_FEATURE_COLUMNS]


# Ключи, таргет и диагностика — в модель не идут
NON_FEATURE_COLUMNS = {
    "id",
    "date",
    "sales",
    "weather_source",
    "days_since_first_sale",
}


def categorical_columns(cfg: Config) -> list[str]:
    """Категориальные колонки для LightGBM.

    Категории фиксируются на train и переиспользуются на inference,
    иначе коды поедут и модель предскажет ерунду, ничего не сломав.
    """
    return list(cfg.features.categorical)


def assert_no_leakage(con, as_of: date | str) -> None:
    """Проверить, что в витрине нет таргета позже as_of.

    Дешёвая проверка, которая ловит самый дорогой класс багов.
    Вызывается после каждой сборки фич, в том числе в проде.
    """
    as_of = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    row = con.execute("SELECT MAX(date) FROM ft.mart WHERE sales IS NOT NULL").fetchone()
    max_target_date = row[0] if row else None

    if max_target_date is not None and max_target_date > as_of:
        raise LeakageError(
            f"В ft.mart есть таргет за {max_target_date}, что позже as_of={as_of}. "
            "Это утечка: фичи посчитаны по данным, которых в момент прогноза не существует."
        )
    logger.info(
        "Проверка утечки пройдена: max(date) с таргетом = %s <= as_of %s", max_target_date, as_of
    )


def inference_grid(cfg: Config, as_of: date | str) -> str:
    """SQL, генерирующий сетку (id × 28 будущих дат) для инференса.

    На inference будущих строк в sales_long ещё нет — их надо создать,
    чтобы было куда джойнить календарь, цены и погодный прогноз.

    TODO: подключить к build_features(mode='inference') вместе с models/predict.py.
    """
    as_of = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    horizon = int(cfg.project.horizon)
    return f"""
        SELECT
            h.id,
            (DATE '{as_of}' + INTERVAL (g.offset) DAY)::DATE AS date,
            g.offset AS horizon
        FROM stg.hierarchy h
        CROSS JOIN generate_series(1, {horizon}) AS g(offset)
    """
