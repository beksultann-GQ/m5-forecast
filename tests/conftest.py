"""Фикстуры. Синтетика вместо реальных 59 млн строк.

Принцип: тесты не должны требовать скачанного датасета и не должны ходить в сеть.
Всё, что проверяется, проверяется на маленьких данных с ИЗВЕСТНЫМ ответом —
только так тест ловит ошибку, а не подтверждает её.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

RNG_SEED = 42
N_ITEMS = 5
N_STORES = 2
N_DAYS = 400
ORIGIN = date(2015, 1, 1)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="session")
def synthetic_sales(rng: np.random.Generator) -> pd.DataFrame:
    """Синтетические продажи с ИЗВЕСТНОЙ структурой.

    Заложено намеренно, чтобы тесты могли это проверить:
        - недельная сезонность (выходные выше)
        - годовой тренд
        - ~60% нулей (как в реальном M5)
        - у части рядов продажи начинаются не с первого дня (новые товары)
    """
    dates = [ORIGIN + timedelta(days=i) for i in range(N_DAYS)]
    rows = []
    for store_i in range(N_STORES):
        store_id = f"CA_{store_i + 1}"
        for item_i in range(N_ITEMS):
            item_id = f"FOODS_1_{item_i + 1:03d}"
            base = rng.uniform(0.5, 8.0)
            # у половины рядов "товар появился" на 100-й день
            start_day = 100 if item_i % 2 else 0
            for day_i, d in enumerate(dates):
                if day_i < start_day:
                    sales = 0
                else:
                    dow_factor = 1.4 if d.weekday() >= 5 else 1.0
                    trend = 1.0 + day_i / N_DAYS * 0.3
                    lam = base * dow_factor * trend
                    sales = int(rng.poisson(lam) * (rng.random() > 0.35))
                rows.append(
                    {
                        "id": f"{item_id}_{store_id}_evaluation",
                        "item_id": item_id,
                        "dept_id": "FOODS_1",
                        "cat_id": "FOODS",
                        "store_id": store_id,
                        "state_id": "CA",
                        "date": pd.Timestamp(d),
                        "sales": sales,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_calendar() -> pd.DataFrame:
    """Календарь на тот же период."""
    dates = [ORIGIN + timedelta(days=i) for i in range(N_DAYS)]
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "d": [f"d_{i + 1}" for i in range(N_DAYS)],
            "wm_yr_wk": [11500 + i // 7 for i in range(N_DAYS)],
            "wday": [(d.weekday() + 2) % 7 + 1 for d in dates],
            "month": [d.month for d in dates],
            "year": [d.year for d in dates],
            "event_name_1": [None] * N_DAYS,
            "event_type_1": [None] * N_DAYS,
            "event_name_2": [None] * N_DAYS,
            "event_type_2": [None] * N_DAYS,
            "snap_CA": [1 if d.day <= 10 else 0 for d in dates],
            "snap_TX": [1 if d.day <= 10 else 0 for d in dates],
            "snap_WI": [1 if d.day <= 10 else 0 for d in dates],
        }
    )


@pytest.fixture(scope="session")
def synthetic_prices(rng: np.random.Generator) -> pd.DataFrame:
    """Недельные цены."""
    rows = []
    for store_i in range(N_STORES):
        for item_i in range(N_ITEMS):
            price = rng.uniform(1.0, 15.0)
            for week in range(N_DAYS // 7 + 1):
                rows.append(
                    {
                        "store_id": f"CA_{store_i + 1}",
                        "item_id": f"FOODS_1_{item_i + 1:03d}",
                        "wm_yr_wk": 11500 + week,
                        "sell_price": round(price * rng.uniform(0.9, 1.1), 2),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def duckdb_con():
    """In-memory DuckDB. Каждый тест получает чистую базу."""
    import duckdb

    con = duckdb.connect(":memory:")
    yield con
    con.close()


@pytest.fixture(scope="session")
def cfg():
    """Конфиг проекта."""
    from m5.config import load_config

    return load_config()


@pytest.fixture
def open_meteo_response() -> dict:
    """Зафиксированный ответ Open-Meteo Archive.

    Записан один раз, лежит в репозитории. Тесты парсинга не ходят в сеть:
    внешний API не должен уметь ронять наш CI.
    """
    return {
        "latitude": 34.05,
        "longitude": -118.24,
        "timezone": "America/Los_Angeles",
        "daily_units": {"time": "iso8601", "temperature_2m_max": "°C"},
        "daily": {
            "time": ["2015-01-01", "2015-01-02", "2015-01-03"],
            "temperature_2m_max": [18.4, 19.1, 21.0],
            "temperature_2m_min": [8.2, 9.0, 10.4],
            "temperature_2m_mean": [13.1, 14.0, 15.7],
            "precipitation_sum": [0.0, 2.3, 0.0],
            "snowfall_sum": [0.0, 0.0, 0.0],
            "windspeed_10m_max": [11.2, 14.8, 9.6],
        },
    }


@pytest.fixture
def nager_response() -> list[dict]:
    """Зафиксированный ответ Nager.Date.

    Namely: Martin Luther King Day — общенациональный (counties=null),
    Cesar Chavez Day — только Калифорния. Второй случай ровно тот,
    ради которого мы вообще берём Nager вместо event_name из M5.
    """
    return [
        {
            "date": "2015-01-01",
            "localName": "New Year's Day",
            "name": "New Year's Day",
            "countryCode": "US",
            "fixed": False,
            "global": True,
            "counties": None,
            "launchYear": None,
            "types": ["Public"],
        },
        {
            "date": "2015-01-19",
            "localName": "Martin Luther King, Jr. Day",
            "name": "Martin Luther King, Jr. Day",
            "countryCode": "US",
            "fixed": False,
            "global": True,
            "counties": None,
            "launchYear": None,
            "types": ["Public"],
        },
        {
            "date": "2015-03-31",
            "localName": "Cesar Chavez Day",
            "name": "Cesar Chavez Day",
            "countryCode": "US",
            "fixed": False,
            "global": False,
            "counties": ["US-CA"],
            "launchYear": None,
            "types": ["Public"],
        },
    ]
