"""Фактическое качество: считаем ошибку, когда приезжают реальные продажи.

Это единственная метрика, которая действительно что-то значит. Всё остальное —
бэктест, валидация, дрифт — прокси. Здесь мы сравниваем прогноз, сделанный
28 дней назад, с тем, что реально продалось.

Результат складываем в mart.forecast_accuracy: со временем накапливается
история качества, по которой видно деградацию и сезонные провалы.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from m5.config import Config
from m5.utils.logging import get_logger

logger = get_logger(__name__)

# Уровни, на которых считаем фактическое качество (упрощённые 12 уровней WRMSSE)
REPORT_LEVELS: list[tuple[str, list[str]]] = [
    ("total", []),
    ("state", ["state_id"]),
    ("store", ["store_id"]),
    ("cat", ["cat_id"]),
    ("dept", ["dept_id"]),
    ("store_dept", ["store_id", "dept_id"]),
    ("item_store", ["item_id", "store_id"]),
]


def compute_actual_accuracy(cfg: Config, as_of: date, window_days: int = 28) -> pd.DataFrame:
    """Посчитать фактическое качество прогнозов, сделанных window_days назад.

    Джойн mart.forecast (as_of = target_date - horizon) с фактом из stg.sales_long.

    Returns:
        Строки для mart.forecast_accuracy по всем REPORT_LEVELS.

    TODO:
        1. вытащить прогнозы, у которых целевая дата уже наступила и факт есть
        2. джойн с фактом
        3. по каждому уровню сагрегировать и посчитать mae/wmape/rmse/bias
        4. записать в mart.forecast_accuracy идемпотентно
    """
    raise NotImplementedError


def accuracy_trend(cfg: Config, level: str = "total", n_periods: int = 12) -> pd.DataFrame:
    """История качества по уровню — для дашборда.

    TODO: SELECT из mart.forecast_accuracy, отсортировать по as_of.
    """
    raise NotImplementedError


def should_retrain(cfg: Config, threshold_pct: float = 15.0) -> tuple[bool, str]:
    """Правило автопереобучения.

    Триггерим, если WMAPE за последние 28 дней выросла больше чем на
    threshold_pct относительно среднего за предыдущие 3 периода.

    Returns:
        (нужно_ли_переобучать, причина_текстом)

    Причина текстом обязательна: она уезжает в алерт, и дежурный должен
    понять, что произошло, не открывая код.

    TODO: реализовать сравнение.
    """
    raise NotImplementedError


def worst_series(cfg: Config, as_of: date, top_n: int = 50) -> pd.DataFrame:
    """Топ худших рядов по вкладу в суммарную ошибку.

    Практический смысл: закупщику интересны не средние 20% ошибки,
    а конкретные позиции, по которым он регулярно промахивается.

    TODO: посчитать вклад каждого ряда в sum|y-yhat|, отсортировать.
    """
    raise NotImplementedError
