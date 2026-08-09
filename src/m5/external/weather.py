"""Погода: Open-Meteo (основной) / Visual Crossing (запасной).

Гипотеза: погода двигает продажи. Жара -> напитки и мороженое, снегопад ->
провал трафика в магазин, дождь -> сдвиг спроса на доставку.
Для FOODS (крупнейшая категория M5) эффект должен быть заметен.

API:
    Archive (история, 1940-настоящее, задержка ~5 дней, ключ не нужен):
        GET https://archive-api.open-meteo.com/v1/archive
            ?latitude=34.05&longitude=-118.24
            &start_date=2011-01-29&end_date=2016-06-19
            &daily=temperature_2m_max,temperature_2m_min,precipitation_sum
            &timezone=America/Los_Angeles
    Forecast (до 16 дней вперёд, ключ не нужен):
        GET https://api.open-meteo.com/v1/forecast?...&forecast_days=16

ГЛАВНАЯ ЛОВУШКА (training/serving skew).
    На train погода известна фактическая. На inference за горизонт 28 дней
    фактической погоды НЕТ и быть не может. Если обучиться на факте, а в проде
    подставлять прогноз погоды — распределение фичи поедет, и модель поедет с ним.
    Стратегия из конфига `future_strategy: forecast_then_climatology`:
        дни 1-16  -> прогноз Open-Meteo
        дни 17-28 -> климатическая норма по дню года (среднее за 2011-2016)
    Честный вариант — обучаться сразу на «прогнозной» погоде, но исторических
    прогнозов у нас нет, поэтому логируем это как известное ограничение
    и обязательно проверяем на бэктесте, что фича вообще даёт прирост.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from m5.config import Config
from m5.external.client import CachedJSONClient
from m5.external.geo import Location
from m5.utils.logging import get_logger

logger = get_logger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
VISUAL_CROSSING_URL = (
    "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
)

# Archive API отдаёт данные с задержкой ~5 дней — за этот хвост берём forecast API
ARCHIVE_LAG_DAYS = 5


# ---------------------------------------------------------------- Open-Meteo


def build_archive_params(
    loc: Location,
    start: date,
    end: date,
    daily_vars: list[str],
    units: dict[str, str] | None = None,
) -> dict[str, object]:
    """Параметры запроса к Open-Meteo Archive."""
    params: dict[str, object] = {
        "latitude": loc.lat,
        "longitude": loc.lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(daily_vars),
        "timezone": loc.tz,
    }
    if units:
        params.update(units)
    return params


def build_forecast_params(
    loc: Location,
    daily_vars: list[str],
    forecast_days: int = 16,
    past_days: int = 7,
    units: dict[str, str] | None = None,
) -> dict[str, object]:
    """Параметры запроса к Open-Meteo Forecast.

    past_days закрывает разрыв между концом archive и сегодня.
    """
    params: dict[str, object] = {
        "latitude": loc.lat,
        "longitude": loc.lon,
        "daily": ",".join(daily_vars),
        "forecast_days": forecast_days,
        "past_days": past_days,
        "timezone": loc.tz,
    }
    if units:
        params.update(units)
    return params


def parse_open_meteo(payload: dict, loc: Location, source: str) -> pd.DataFrame:
    """Ответ Open-Meteo -> DataFrame.

    Формат ответа: {"daily": {"time": [...], "temperature_2m_max": [...], ...}}

    TODO:
        1. pd.DataFrame(payload['daily'])
        2. переименовать time -> date, привести к datetime64
        3. добавить state_id=loc.state_id, city, source
        4. упасть с понятной ошибкой, если 'daily' нет в ответе
    """
    raise NotImplementedError


def fetch_archive(cfg: Config, client: CachedJSONClient) -> pd.DataFrame:
    """Историческая погода по всем точкам из конфига.

    TODO: по каждой Location собрать build_archive_params -> client.get -> parse,
          сконкатенировать, вернуть.
    """
    raise NotImplementedError


def fetch_forecast(cfg: Config, client: CachedJSONClient) -> pd.DataFrame:
    """Прогноз погоды на 16 дней вперёд (для inference).

    Кешировать НЕЛЬЗЯ надолго: прогноз обновляется. use_cache=False.

    TODO: как fetch_archive, но forecast-эндпоинт и use_cache=False.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- Visual Crossing


def build_visual_crossing_params(cfg: Config, loc: Location, start: date, end: date) -> dict:
    """Параметры Visual Crossing (нужен API-ключ, есть бесплатный лимит 1000 записей/день).

    URL: {base}/{lat},{lon}/{start}/{end}?unitGroup=metric&include=days&key=...

    TODO: собрать params; ключ брать из env VISUAL_CROSSING_API_KEY.
    """
    raise NotImplementedError


def parse_visual_crossing(payload: dict, loc: Location) -> pd.DataFrame:
    """Ответ Visual Crossing -> тот же DataFrame, что и у Open-Meteo.

    Колонки маппим в схему Open-Meteo (tempmax -> temperature_2m_max и т.д.),
    чтобы дальше по пайплайну провайдер был не важен.

    TODO: реализовать маппинг колонок.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- производные


def add_derived_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Производные погодные признаки.

    Сырая температура сама по себе слабый признак — важно ОТКЛОНЕНИЕ от нормы:
    +25C в январе в Висконсине и +25C в июле в Калифорнии значат разное.

    Считаем:
        temp_anomaly_vs_doy_norm — темп. минус норма по дню года (по всей истории)
        hdd / cdd                — градусо-дни отопления/охлаждения, база 18C
        is_heavy_rain            — осадки выше p90 по штату
        is_snow_day              — snowfall_sum > 0
        temp_change_1d           — дельта к вчера (резкие перепады -> всплески покупок)

    TODO: реализовать по списку cfg.external.weather.derived.
    """
    raise NotImplementedError


def climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Климатическая норма: среднее по (state_id, dayofyear) за всю историю.

    Используется двояко:
        1. как базис для temp_anomaly
        2. как заполнитель погоды на горизонте 17-28 дней, где прогноза нет

    TODO: сгладить скользящим окном ±7 дней, иначе норма шумная.
    """
    raise NotImplementedError


def extend_with_climatology(df: pd.DataFrame, until: date, cfg: Config) -> pd.DataFrame:
    """Дотянуть погодный ряд до `until` климатической нормой.

    TODO: сгенерировать недостающие даты, подставить норму, source='climatology'.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- точка входа


def fetch_weather(cfg: Config, mode: str = "archive", offline: bool = False) -> Path:
    """Собрать погодную таблицу и положить в data/external/weather.parquet.

    Args:
        mode: 'archive' — историческая (для train)
              'forecast' — прогноз (для ежедневного inference)
              'full' — archive + forecast + климатология до горизонта

    Returns:
        Путь к parquet со схемой WEATHER_SCHEMA.

    TODO:
        1. client = CachedJSONClient(cfg.external.cache_dir, offline=offline)
        2. по mode собрать нужные куски и склеить (приоритет archive > forecast > climatology)
        3. add_derived_features()
        4. validate(df, 'weather')
        5. записать parquet, вернуть путь
    """
    raise NotImplementedError
