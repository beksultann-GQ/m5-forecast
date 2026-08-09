"""Мониторинг дрифта: данных и прогнозов.

Модель не ломается с грохотом — она тихо деградирует, когда мир под ней меняется.
Три вещи, которые надо ловить:

    data drift       — распределение фич уехало относительно обучающей выборки
    prediction drift — прогнозы поехали, хотя фичи вроде те же
    concept drift    — связь фич и таргета изменилась (видно только по факту)

Отдельная тонкость для этого проекта: погодные фичи на инференсе приходят
из forecast/climatology, а на train были archive. Это гарантированный источник
дрифта по построению — за ним надо следить прицельно, по колонке weather_source.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from m5.config import Config
from m5.utils.logging import get_logger

logger = get_logger(__name__)


def reference_sample(cfg: Config, n_rows: int = 200_000) -> pd.DataFrame:
    """Эталонная выборка — срез обучающих данных, с которым сравниваем прод.

    Сохраняется вместе с моделью: сравнивать надо именно с тем, на чём училась
    ТЕКУЩАЯ прод-модель, а не с последним train вообще.

    TODO: сэмплировать ft.mart за окно train, сохранить в артефакты модели.
    """
    raise NotImplementedError


def data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg: Config,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Отчёт по дрифту фич (Evidently).

    Returns:
        {'n_drifted': int, 'share_drifted': float, 'drifted_columns': [...]}

    TODO: Report(metrics=[DataDriftPreset()]), сохранить html в out_path,
          вернуть сводку as_dict.
    """
    raise NotImplementedError


def prediction_drift(cfg: Config, window_days: int = 28) -> dict[str, float]:
    """Дрифт распределения прогнозов: среднее, медиана, p95, доля нулей.

    Дешёвый и быстрый сигнал: если модель вдруг стала предсказывать
    в полтора раза больше, это видно раньше, чем приедет факт.

    TODO: сравнить последние window_days прогнозов из mart.forecast
          с предыдущим окном; вернуть дельты.
    """
    raise NotImplementedError


def weather_source_mix(cfg: Config, as_of: date) -> dict[str, float]:
    """Доли archive / forecast / climatology в погодных фичах прогноза.

    Прицельный мониторинг под наш внешний источник: если доля climatology
    подскочила (Open-Meteo лёг, данные не доехали) — качество на дальнем
    горизонте просядет, и знать об этом надо ДО того, как приедет факт.

    TODO: сгруппировать ft.weather по weather_source за окно прогноза.
    """
    raise NotImplementedError


def external_data_freshness(cfg: Config) -> dict[str, Any]:
    """Свежесть внешних источников: max(date) в weather и holidays.

    Отдельная проверка, потому что внешние API отваливаются независимо
    от нашего пайплайна, и падать надо явно, а не подставлять NULL молча.

    TODO: вернуть {'weather_max_date':..., 'weather_lag_days':...,
                   'holidays_max_year':..., 'is_stale': bool}
    """
    raise NotImplementedError


def check_drift_and_alert(cfg: Config, threshold_share: float = 0.3) -> bool:
    """Проверить дрифт и решить, нужен ли алерт/переобучение.

    Returns:
        True, если сработало правило переобучения.

    TODO: собрать data_drift + prediction_drift + freshness,
          сравнить с порогами, вернуть решение и записать его в лог.
    """
    raise NotImplementedError
