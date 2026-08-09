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
        "sales": Column(int, Check.ge(0), nullable=False),
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
        sample: проверять не весь фрейм, а случайную выборку (59 млн строк
            прогонять через pandera на каждом запуске дорого).
        lazy: собрать все ошибки разом, а не падать на первой.

    Raises:
        DataValidationError: с человекочитаемым отчётом.

    TODO: обернуть schema.validate(df, lazy=lazy) и переупаковать
          pa.errors.SchemaErrors в DataValidationError с топ-10 нарушений.
    """
    raise NotImplementedError


def validate_duckdb_table(con, table: str, schema_name: str, sample_size: int = 500_000) -> dict:
    """Проверить таблицу DuckDB, не вытаскивая её целиком в память.

    Стратегия: часть проверок делаем SQL-запросами прямо в DuckDB
    (дубли, отрицательные значения, дыры в датах), а pandera гоняем
    по случайной выборке — на типы и диапазоны этого хватает.

    TODO:
        1. прогнать sql_checks() -> нарушения
        2. con.execute(f"SELECT * FROM {table} USING SAMPLE {sample_size} ROWS").df()
        3. validate(sample_df, schema_name)
    """
    raise NotImplementedError


def sql_checks(con, table: str) -> dict[str, Any]:
    """Тяжёлые инвариантные проверки — на SQL, а не на pandas.

    Проверяем:
        - нет дублей по (id, date)
        - нет sales < 0
        - календарь непрерывен: COUNT(DISTINCT date) == datediff+1
        - у каждой пары (item, store) в ассортименте есть цена
        - max(date) не в будущем относительно as_of

    TODO: вернуть {check_name: n_violations}, пустые = ок.
    """
    raise NotImplementedError
