"""Метрики. WRMSSE — метрика соревнования, WMAPE/MAE — то, что понимает бизнес.

WRMSSE устроен так (и это надо уметь объяснить вслух, а не просто скопировать):

    1. Ряды агрегируются по 12 уровням: от «все продажи целиком» (уровень 1)
       до «конкретный товар в конкретном магазине» (уровень 12). Всего 42840 рядов.
    2. Для каждого ряда считается RMSSE — RMSE, нормированный на ошибку
       наивного прогноза (продажи = вчера) на обучающем отрезке:

           RMSSE = sqrt( mean((y - yhat)^2) / mean((y_t - y_{t-1})^2) )

       Знаменатель убирает влияние масштаба: ряд с продажами 1000 шт/день
       и ряд с 2 шт/день вносят сопоставимый вклад.
    3. Ряды взвешиваются долей в ВЫРУЧКЕ за последние 28 дней train
       (не в штуках!). Дорогие товары важнее.
    4. WRMSSE = sum(w_i * RMSSE_i), веса внутри каждого уровня суммируются в 1,
       уровни усредняются с равным весом (1/12).

Ключевая деталь, на которой все спотыкаются: знаменатель RMSSE считается
ТОЛЬКО по обучающему отрезку и только начиная с первой ненулевой продажи ряда.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from m5.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------- простые метрики


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(diff**2)))


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE = sum|y - yhat| / sum|y|.

    Почему не MAPE: в M5 ~68% значений нулевые, деление на ноль убивает MAPE.
    WMAPE делит на сумму, а не поштучно — работает и на прерывистом спросе.
    Это та метрика, которую понимает отдел закупок: «мы ошибаемся на 25% объёма».
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Смещение: систематически перепрогнозируем или недопрогнозируем.

    Для закупок важнее знака, чем величины: постоянный недопрогноз = out-of-stock,
    постоянный перепрогноз = замороженные деньги и списания.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(y_true)
    if denom == 0:
        return float("nan")
    return float(np.sum(y_pred - y_true) / denom)


# ---------------------------------------------------------------- WRMSSE

# 12 уровней агрегации из правил соревнования
AGGREGATION_LEVELS: list[list[str]] = [
    [],  # 1  — всё целиком (1 ряд)
    ["state_id"],  # 2  — 3 ряда
    ["store_id"],  # 3  — 10
    ["cat_id"],  # 4  — 3
    ["dept_id"],  # 5  — 7
    ["state_id", "cat_id"],  # 6  — 9
    ["state_id", "dept_id"],  # 7  — 21
    ["store_id", "cat_id"],  # 8  — 30
    ["store_id", "dept_id"],  # 9  — 70
    ["item_id"],  # 10 — 3049
    ["item_id", "state_id"],  # 11 — 9147
    ["item_id", "store_id"],  # 12 — 30490
]


@dataclass
class WRMSSEWeights:
    """Предрассчитанные веса и знаменатели. Считаются ОДИН раз по train.

    Пересчитывать их на каждом фолде дорого (42840 рядов × вся история),
    поэтому кешируем и переиспользуем.
    """

    scales: dict[int, pd.Series]  # уровень -> знаменатель RMSSE по рядам
    weights: dict[int, pd.Series]  # уровень -> вес ряда (доля выручки)

    def save(self, path) -> None:
        """TODO: сохранить в parquet/pickle рядом с артефактами модели."""
        raise NotImplementedError

    @classmethod
    def load(cls, path) -> WRMSSEWeights:
        """TODO: загрузить."""
        raise NotImplementedError


def compute_weights(
    train_sales: pd.DataFrame,
    prices: pd.DataFrame,
    weight_window_days: int = 28,
) -> WRMSSEWeights:
    """Посчитать веса и знаменатели по обучающему отрезку.

    Args:
        train_sales: long-формат (id, item_id, ..., date, sales) до конца train
        prices: (store_id, item_id, wm_yr_wk, sell_price)
        weight_window_days: окно для весов — последние 28 дней train

    Веса = доля ряда в суммарной ВЫРУЧКЕ (sales * sell_price) за окно.
    Знаменатель = mean((y_t - y_{t-1})^2) по train, считая с первой ненулевой продажи.

    TODO:
        1. посчитать выручку по каждому ряду за последние weight_window_days
        2. для каждого уровня агрегации: сгруппировать, нормировать веса к сумме 1
        3. для каждого уровня: агрегировать ряды, посчитать средний квадрат
           первой разности, обрезав ведущие нули
        4. вернуть WRMSSEWeights
    """
    raise NotImplementedError


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, scale: float) -> float:
    """RMSSE одного ряда при известном знаменателе."""
    if scale <= 0 or np.isnan(scale):
        return float("nan")
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2) / scale))


def wrmsse(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    weights: WRMSSEWeights,
    return_by_level: bool = False,
) -> float | tuple[float, dict[int, float]]:
    """Weighted Root Mean Squared Scaled Error по 12 уровням.

    Args:
        y_true: факт, long-формат с колонками иерархии + date + sales
        y_pred: прогноз в том же формате с колонкой prediction
        weights: результат compute_weights по соответствующему train
        return_by_level: вернуть ещё и разбивку по уровням — это диагностика,
            показывающая, где именно модель плоха (на топовых агрегатах
            или на отдельных товарах)

    TODO:
        1. смёржить факт и прогноз по (id, date), проверить, что ничего не потерялось
        2. по каждому уровню: агрегировать sum, посчитать rmsse по рядам
        3. взвесить, просуммировать внутри уровня
        4. усреднить 12 уровней с равным весом
    """
    raise NotImplementedError


def metrics_report(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
    weights: WRMSSEWeights | None = None,
) -> dict[str, float]:
    """Все метрики разом — то, что уходит в MLflow.

    TODO: собрать {'wrmsse':..., 'wmape':..., 'mae':..., 'rmse':..., 'bias':...}
          плюс разбивку по горизонтам (h1-7, h8-14, h15-21, h22-28):
          качество на дальнем горизонте всегда хуже, и это надо видеть.
    """
    raise NotImplementedError
