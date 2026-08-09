"""Baseline. Считается ПЕРВЫМ, до всякого ML.

Без baseline любой ML — карго-культ: непонятно, что значит WMAPE 0.42,
хорошо это или стыдно. Это первое, что спросят на собеседовании.

Три baseline'а:
    seasonal_naive — продажи = ровно 28 дней назад (главный, на него нормирован RMSSE)
    moving_average — среднее за последние 28 дней
    dow_mean       — среднее по этому дню недели за 8 недель

dow_mean — самый честный ориентир: он уже умеет недельную сезонность,
и если LightGBM не бьёт его, значит модель не выучила ничего сверх календаря.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from m5.config import Config
from m5.data.access import data_bounds, load_sales
from m5.evaluation.cv import Fold, make_folds
from m5.evaluation.metrics import bias, mae, rmse, wmape
from m5.utils.logging import get_logger

logger = get_logger(__name__)


def seasonal_naive(
    history: pd.DataFrame,
    as_of: date | pd.Timestamp,
    horizon: int = 28,
    lag_days: int = 28,
) -> pd.DataFrame:
    """Прогноз = продажи ровно `lag_days` назад.

    Почему именно 28: горизонт равен 28, значит это минимальный лаг, доступный
    на инференсе. Он же кратен 7, поэтому сохраняет день недели.

    Args:
        history: long-формат (id, date, sales) до as_of включительно
        as_of: граница знания — прогнозируем as_of+1 .. as_of+horizon

    Returns:
        (id, date, horizon, prediction)
    """
    as_of = pd.Timestamp(as_of)
    window_start = as_of - pd.Timedelta(days=lag_days - 1)

    window = history.loc[
        history["date"].between(window_start, as_of), ["id", "date", "sales"]
    ].copy()
    window["date"] = window["date"] + pd.Timedelta(days=lag_days)
    window["horizon"] = (window["date"] - as_of).dt.days
    window = window[window["horizon"].between(1, horizon)]

    return window.rename(columns={"sales": "prediction"})[["id", "date", "horizon", "prediction"]]


def moving_average(
    history: pd.DataFrame,
    as_of: date | pd.Timestamp,
    horizon: int = 28,
    window: int = 28,
) -> pd.DataFrame:
    """Прогноз = среднее за последние `window` дней, одинаковое на весь горизонт."""
    as_of = pd.Timestamp(as_of)
    recent = history.loc[history["date"].between(as_of - pd.Timedelta(days=window - 1), as_of)]
    level = recent.groupby("id", observed=True)["sales"].mean().rename("prediction")

    return _broadcast_over_horizon(level, as_of, horizon)


def dow_mean(
    history: pd.DataFrame,
    as_of: date | pd.Timestamp,
    horizon: int = 28,
    n_weeks: int = 8,
) -> pd.DataFrame:
    """Прогноз = среднее по этому дню недели за последние n_weeks недель.

    Сильнее seasonal_naive: усредняет шум, но сохраняет недельную сезонность.
    """
    as_of = pd.Timestamp(as_of)
    recent = history.loc[
        history["date"].between(as_of - pd.Timedelta(days=n_weeks * 7 - 1), as_of)
    ].copy()
    recent["dow"] = recent["date"].dt.dayofweek
    level = recent.groupby(["id", "dow"], observed=True)["sales"].mean().rename("prediction")

    grid = _horizon_grid(history["id"].unique(), as_of, horizon)
    grid["dow"] = grid["date"].dt.dayofweek
    result = grid.merge(level.reset_index(), on=["id", "dow"], how="left")
    result["prediction"] = result["prediction"].fillna(0.0)

    return result[["id", "date", "horizon", "prediction"]]


BASELINES = {
    "seasonal_naive": seasonal_naive,
    "moving_average": moving_average,
    "dow_mean": dow_mean,
}


def run_baselines(cfg: Config, log_to_mlflow: bool = False) -> pd.DataFrame:
    """Посчитать все baseline'ы на всех фолдах.

    Число, записанное здесь, — планка на весь проект.

    Returns:
        Таблица (baseline × фолд × метрика).
    """
    data_start, data_end = data_bounds(cfg)
    folds = make_folds(cfg, data_start, data_end)
    horizon = int(cfg.project.horizon)

    rows = []
    for fold in folds:
        history, actual = _fold_data(cfg, fold)
        for name, fn in BASELINES.items():
            preds = fn(history, fold.forecast_origin, horizon=horizon)
            metrics = score(actual, preds)
            rows.append({"baseline": name, "fold": fold.index, **metrics})
            logger.info(
                "fold=%d %-15s wmape=%.4f mae=%.4f bias=%+.4f",
                fold.index,
                name,
                metrics["wmape"],
                metrics["mae"],
                metrics["bias"],
            )

    frame = pd.DataFrame(rows)
    summary = frame.groupby("baseline")[["wmape", "mae", "rmse", "bias"]].mean()
    logger.info("Среднее по фолдам:\n%s", summary.to_string())
    return frame


def score(actual: pd.DataFrame, preds: pd.DataFrame) -> dict[str, float]:
    """Сравнить прогноз с фактом по ключу (id, date).

    Джойн внутренний, но с проверкой: если прогноз покрыл не все факты,
    метрика посчитается по подмножеству и будет молча оптимистичной.
    """
    merged = actual.merge(preds, on=["id", "date"], how="left", validate="one_to_one")
    n_missing = int(merged["prediction"].isna().sum())
    if n_missing:
        logger.warning(
            "Нет прогноза для %d из %d строк факта — заполняю нулями", n_missing, len(merged)
        )
        merged["prediction"] = merged["prediction"].fillna(0.0)

    y_true = merged["sales"].to_numpy(dtype=float)
    y_pred = merged["prediction"].to_numpy(dtype=float)
    return {
        "wmape": wmape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "n_rows": len(merged),
    }


def _fold_data(cfg: Config, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    """История до forecast_origin и факт на валидационном окне.

    Грузим не всю историю, а только нужное окно: baseline'ам хватает
    последних ~60 дней, а таскать 59 млн строк ради среднего — расточительство.
    """
    lookback = pd.Timestamp(fold.forecast_origin) - pd.Timedelta(days=90)
    history = load_sales(cfg, start=lookback.date(), end=fold.forecast_origin)
    actual = load_sales(cfg, start=fold.val_start, end=fold.val_end)
    return history, actual


def _horizon_grid(ids, as_of: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Сетка (id × горизонты 1..horizon) с датами."""
    horizons = range(1, horizon + 1)
    grid = pd.MultiIndex.from_product([ids, horizons], names=["id", "horizon"]).to_frame(
        index=False
    )
    grid["date"] = as_of + pd.to_timedelta(grid["horizon"], unit="D")
    return grid


def _broadcast_over_horizon(level: pd.Series, as_of: pd.Timestamp, horizon: int) -> pd.DataFrame:
    """Один уровень на ряд -> строки на весь горизонт."""
    grid = _horizon_grid(level.index.to_numpy(), as_of, horizon)
    result = grid.merge(level.reset_index(), on="id", how="left")
    result["prediction"] = result["prediction"].fillna(0.0)
    return result[["id", "date", "horizon", "prediction"]]
