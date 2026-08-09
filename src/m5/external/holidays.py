"""Госпраздники США: Nager.Date.

Зачем, если в calendar.csv уже есть event_name_1/2:
    1. В M5 события — плоский список без региональности. Nager отдаёт праздники
       по субдивизиям (US-CA / US-TX / US-WI), а Cesar Chavez Day, например,
       выходной в Калифорнии и обычный день в Техасе.
    2. Nager различает public / bank / school / optional — разный эффект на трафик.
    3. Независимый источник = можно свериться и найти дыры в M5-событиях.
    4. На inference календарь праздников известен наперёд на годы вперёд.
       В отличие от погоды, здесь НЕТ проблемы training/serving skew — это
       идеальная фича: одинаково доступна и на train, и в проде.

API (без ключа, без лимитов в разумных пределах):
    GET https://date.nager.at/api/v3/PublicHolidays/{year}/{countryCode}
    -> [{"date":"2011-01-17","localName":"Martin Luther King, Jr. Day",
         "name":"Martin Luther King, Jr. Day","countryCode":"US",
         "fixed":false,"global":true,"counties":["US-CA","US-TX"],
         "launchYear":null,"types":["Public"]}]

    Поле counties: null = праздник по всей стране, список = только эти штаты.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from m5.config import Config
from m5.external.client import CachedJSONClient
from m5.utils.logging import get_logger

logger = get_logger(__name__)

NAGER_BASE_URL = "https://date.nager.at/api/v3"


def public_holidays_url(year: int, country_code: str = "US") -> str:
    """Эндпоинт праздников за год."""
    return f"{NAGER_BASE_URL}/PublicHolidays/{year}/{country_code}"


def long_weekend_url(year: int, country_code: str = "US") -> str:
    """Длинные выходные (праздник + примыкающие выходные).

    Отдельный полезный эндпоинт: длинный уикенд — это поездки и закупка впрок,
    эффект на продажи сильнее одиночного праздника.
    """
    return f"{NAGER_BASE_URL}/LongWeekend/{year}/{country_code}"


def parse_holidays(payload: list[dict], subdivisions: list[str]) -> pd.DataFrame:
    """Ответ Nager.Date -> длинный DataFrame (date, subdivision, holiday_name, ...).

    Логика раскрытия counties:
        counties == null  -> праздник действует во всех subdivisions
        counties == [...] -> только в пересечении с нашими subdivisions

    Итоговые колонки:
        date, country_code, subdivision, holiday_name, holiday_type,
        is_public_holiday, is_global, is_fixed

    TODO: реализовать разворот, привести date к datetime64.
    """
    raise NotImplementedError


def fetch_raw_holidays(cfg: Config, client: CachedJSONClient) -> pd.DataFrame:
    """Скачать праздники за все годы из конфига.

    Годы бесконечно кешируем: календарь 2011 года уже не изменится.

    TODO: по годам years_from..years_to -> client.get(public_holidays_url(...))
          -> parse_holidays -> concat.
    """
    raise NotImplementedError


def fetch_long_weekends(cfg: Config, client: CachedJSONClient) -> pd.DataFrame:
    """Скачать длинные выходные, развернуть в (date, is_long_weekend).

    TODO: по каждому диапазону startDate..endDate сгенерировать даты.
    """
    raise NotImplementedError


def add_derived_features(holidays: pd.DataFrame, calendar_dates: pd.DataFrame) -> pd.DataFrame:
    """Развернуть праздники в дневной календарь по штатам.

    Сам факт «сегодня праздник» — слабый признак. Работают окрестности:
    закупаются НАКАНУНЕ, а в сам день магазин часто закрыт или полупустой.

    Считаем на сетке (date × state_id):
        is_holiday, is_public_holiday, holiday_name
        is_day_before_holiday, is_day_after_holiday
        days_to_next_holiday, days_since_prev_holiday
        is_holiday_week (в пределах ±3 дней)
        is_long_weekend

    TODO: cross join дат и штатов, left join праздников, оконные функции
          для расстояний до ближайшего праздника.
    """
    raise NotImplementedError


def compare_with_m5_events(holidays: pd.DataFrame, m5_calendar: pd.DataFrame) -> pd.DataFrame:
    """Сверка Nager.Date с event_name_1/2 из M5. Диагностика, не фича.

    Что показывает:
        - какие праздники Nager знает, а M5 нет (и наоборот)
        - какие праздники M5 даёт глобально, а они на деле региональные

    Результат стоит положить в README: это ровно то «а зачем ты вообще
    брал внешние данные», о чём спросят на собеседовании.

    TODO: вернуть таблицу (date, in_nager, in_m5, name_nager, name_m5).
    """
    raise NotImplementedError


def fetch_holidays(cfg: Config, offline: bool = False) -> Path:
    """Собрать таблицу праздников -> data/external/holidays.parquet.

    Returns:
        Путь к parquet со схемой HOLIDAYS_SCHEMA.

    TODO:
        1. client = CachedJSONClient(cfg.external.cache_dir, offline=offline)
        2. fetch_raw_holidays + fetch_long_weekends
        3. add_derived_features на сетке дат календаря M5
        4. validate(df, 'holidays')
        5. записать parquet
    """
    raise NotImplementedError
