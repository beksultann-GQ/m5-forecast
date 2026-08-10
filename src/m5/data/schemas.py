"""Схемы данных (pandera). Пайплайн должен ПАДАТЬ на битых данных.

Проверки навешиваем на границах слоёв: raw -> stg -> ft -> mart.
Молча учить модель на мусоре — худший из возможных исходов.
"""

from __future__ import annotations

from typing import Any

from pandera.pandas import Check, Column, DataFrameSchema

from m5.utils.logging import get_logger

logger = get_logger(__name__)

STATES = ["CA", "TX", "WI"]
CATEGORIES = ["HOBBIES", "HOUSEHOLD", "FOODS"]

# ---------------------------------------------------------------- raw-слой

CALENDAR_SCHEMA = DataFrameSchema(
    {
        "date": Column("datetime64[ns]", unique=True, nullable=False),
        "wm_yr_wk": Column(int, Check.in_range(11101, 11621), nullable=False),
        "wday": Column(int, Check.in_range(1, 7), nullable=False),
        "month": Column(int, Check.in_range(1, 12), nullable=False),
        "year": Column(int, Check.in_range(2011, 2017), nullable=False),
        "d": Column(str, Check.str_matches(r"^d_\d+$"), unique=True, nullable=False),
        "event_name_1": Column(str, nullable=True),
        "event_type_1": Column(str, nullable=True),
        "event_name_2": Column(str, nullable=True),
        "event_type_2": Column(str, nullable=True),
        "snap_CA": Column(int, Check.isin([0, 1])),
        "snap_TX": Column(int, Check.isin([0, 1])),
        "snap_WI": Column(int, Check.isin([0, 1])),
    },
    strict=False,
    coerce=True,
    name="calendar",
)

PRICES_SCHEMA = DataFrameSchema(
    {
        "store_id": Column(str, nullable=False),
        "item_id": Column(str, nullable=False),
        "wm_yr_wk": Column(int, nullable=False),
        "sell_price": Column(float, Check.gt(0), nullable=False),
    },
    unique=["store_id", "item_id", "wm_yr_wk"],
    strict=False,
    coerce=True,
    name="sell_prices",
)

# ---------------------------------------------------------------- stg-слой

SALES_LONG_SCHEMA = DataFrameSchema(
    {
        "id": Column(str, nullable=False),
        "item_id": Column(str, nullable=False),
        "dept_id": Column(str, nullable=False),
        "cat_id": Column(str, Check.isin(CATEGORIES), nullable=False),
        "store_id": Column(str, nullable=False),
        "state_id": Column(str, Check.isin(STATES), nullable=False),
        "date": Column("datetime64[ns]", nullable=False),
        # nullable по построению: staging строит полную сетку по календарю,
        # а календарь длиннее продаж на горизонт. У будущих дней факта нет.
        # Что NULL'ов нет в ПРОШЛОМ — проверяется SQL-инвариантом no_missing_sales_in_past.
        "sales": Column(float, Check.ge(0), nullable=True),
    },
    unique=["id", "date"],
    strict=False,
    coerce=True,
    name="sales_long",
)

# ---------------------------------------------------------------- ft-слой

FEATURES_SCHEMA = DataFrameSchema(
    {
        "id": Column(str, nullable=False),
        "date": Column("datetime64[ns]", nullable=False),
        "sales": Column(float, Check.ge(0), nullable=True),  # на inference таргета нет
        "sell_price": Column(float, Check.gt(0), nullable=True),
    },
    unique=["id", "date"],
    strict=False,
    coerce=True,
    name="features",
)

PREDICTIONS_SCHEMA = DataFrameSchema(
    {
        "id": Column(str, nullable=False),
        "date": Column("datetime64[ns]", nullable=False),
        "horizon": Column(int, Check.in_range(1, 28), nullable=False),
        "prediction": Column(float, Check.ge(0), nullable=False),
        "model_version": Column(str, nullable=False),
        "run_id": Column(str, nullable=False),
    },
    unique=["id", "date"],
    strict=False,
    coerce=True,
    name="predictions",
)

# ---------------------------------------------------------------- внешние данные

WEATHER_SCHEMA = DataFrameSchema(
    {
        "state_id": Column(str, Check.isin(STATES), nullable=False),
        "date": Column("datetime64[ns]", nullable=False),
        "temperature_2m_mean": Column(float, Check.in_range(-60, 60), nullable=True),
        "precipitation_sum": Column(float, Check.ge(0), nullable=True),
        "source": Column(str, nullable=False),  # archive | forecast | climatology
    },
    unique=["state_id", "date"],
    strict=False,
    coerce=True,
    name="weather",
)

HOLIDAYS_SCHEMA = DataFrameSchema(
    {
        "date": Column("datetime64[ns]", nullable=False),
        "country_code": Column(str, nullable=False),
        "subdivision": Column(str, nullable=True),
        "holiday_name": Column(str, nullable=False),
        "is_public_holiday": Column(bool, nullable=False),
    },
    strict=False,
    coerce=True,
    name="holidays",
)

SCHEMAS: dict[str, DataFrameSchema] = {
    "calendar": CALENDAR_SCHEMA,
    "sell_prices": PRICES_SCHEMA,
    "sales_long": SALES_LONG_SCHEMA,
    "features": FEATURES_SCHEMA,
    "predictions": PREDICTIONS_SCHEMA,
    "weather": WEATHER_SCHEMA,
    "holidays": HOLIDAYS_SCHEMA,
}


class DataValidationError(RuntimeError):
    """Данные не прошли проверку — пайплайн останавливаем."""


def validate(df, schema_name: str, sample: int | None = None, lazy: bool = True):
    """Проверить датафрейм по именованной схеме.

    Args:
        sample: проверять не весь фрейм, а случайную выборку. 59 млн строк
            гонять через pandera на каждом запуске — минуты впустую;
            на типы и диапазоны выборки хватает.
        lazy: собрать все ошибки разом, а не падать на первой — иначе
            чинишь по одной и перезапускаешь пайплайн десять раз.

    Raises:
        DataValidationError: с человекочитаемым отчётом.
    """
    import pandera.errors as pa_errors

    if schema_name not in SCHEMAS:
        raise KeyError(f"Нет схемы {schema_name!r}. Доступны: {sorted(SCHEMAS)}")

    frame = df
    if sample and len(df) > sample:
        frame = df.sample(n=sample, random_state=42)
        logger.info("Проверяю выборку %s из %s строк", f"{sample:,}", f"{len(df):,}")

    try:
        SCHEMAS[schema_name].validate(frame, lazy=lazy)
    except pa_errors.SchemaErrors as exc:
        raise DataValidationError(_format_schema_errors(exc, schema_name)) from exc
    except pa_errors.SchemaError as exc:
        raise DataValidationError(f"Схема {schema_name}: {exc}") from exc

    return frame


def _format_schema_errors(exc, schema_name: str) -> str:
    """Отчёт об ошибках схемы: что именно и сколько раз."""
    failures = exc.failure_cases
    total = len(failures)
    head = failures.head(10).to_string(index=False)
    return (
        f"Данные не прошли схему {schema_name}: {total} нарушений.\n"
        f"Первые 10:\n{head}\n"
        f"Пайплайн остановлен намеренно — обучаться на битых данных нельзя."
    )


def validate_duckdb_table(
    con,
    table: str,
    schema_name: str,
    sample_size: int = 200_000,
    fail_on_error: bool = True,
) -> dict[str, Any]:
    """Проверить таблицу DuckDB, не вытаскивая её целиком в память.

    Двухуровневая стратегия:
        1. инвариантные проверки — SQL'ом прямо в DuckDB (дубли, отрицательные
           значения, дыры в датах): считаются на всей таблице и почти бесплатно
        2. типы и диапазоны — pandera по случайной выборке

    Returns:
        {'table', 'n_rows', 'sql_checks', 'schema_ok'}

    Raises:
        DataValidationError: если fail_on_error и есть нарушения.
    """
    n_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    violations = sql_checks(con, table)
    failed = {name: count for name, count in violations.items() if count}

    for name, count in violations.items():
        logger.info("  %s %-32s нарушений: %s", "✗" if count else "✓", name, f"{count:,}")

    if failed and fail_on_error:
        raise DataValidationError(
            f"Таблица {table} не прошла инвариантные проверки: {failed}.\n"
            f"Пайплайн остановлен: молча учить модель на битых данных хуже, чем упасть."
        )

    sample = con.execute(f"SELECT * FROM {table} USING SAMPLE {sample_size} ROWS").df()
    validate(sample, schema_name)

    return {
        "table": table,
        "n_rows": n_rows,
        "sql_checks": violations,
        "schema_ok": True,
    }


def sql_checks(con, table: str) -> dict[str, int]:
    """Инвариантные проверки на SQL, а не на pandas.

    Считаются на всей таблице средствами DuckDB — это дёшево, в отличие
    от выгрузки десятков миллионов строк в память.

    Returns:
        {имя_проверки: число_нарушений}. Нули = всё хорошо.
    """
    columns = {row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()}
    checks: dict[str, int] = {}

    def scalar(query: str) -> int:
        row = con.execute(query).fetchone()
        return int(row[0] or 0) if row else 0

    if {"id", "date"} <= columns:
        checks["дубли по (id, date)"] = scalar(
            f"SELECT COUNT(*) FROM (SELECT id, date FROM {table} GROUP BY 1, 2 HAVING COUNT(*) > 1)"
        )

    if "sales" in columns:
        checks["отрицательные продажи"] = scalar(f"SELECT COUNT(*) FROM {table} WHERE sales < 0")

        # NULL допустим только на будущих датах (их календарь знает, а факта ещё нет).
        # NULL в прошлом означает потерянные строки при джойне.
        checks["пропуски факта в прошлом"] = scalar(
            f"""
            SELECT COUNT(*) FROM {table}
            WHERE sales IS NULL
              AND date <= (SELECT MAX(date) FROM {table} WHERE sales IS NOT NULL)
            """
        )

    if "date" in columns:
        # Дыра в календаре молча сдвигает все лаги на этом ряду
        checks["дыры в календаре"] = scalar(
            f"""
            SELECT DATEDIFF('day', MIN(date), MAX(date)) + 1 - COUNT(DISTINCT date)
            FROM {table}
            """
        )

    if {"sell_price", "sales"} <= columns:
        # Товар продаётся, но цены на эту неделю нет — дыра в справочнике
        checks["продажи без цены"] = scalar(
            f"SELECT COUNT(*) FROM {table} WHERE sales > 0 AND sell_price IS NULL"
        )

    if "sell_price" in columns:
        checks["нулевая или отрицательная цена"] = scalar(
            f"SELECT COUNT(*) FROM {table} WHERE sell_price <= 0"
        )

    return checks
