"""Бэктест: прогнать модель по всем фолдам и собрать честные метрики.

Отличие от «посчитал скор на валидации»: здесь на каждом фолде заново
пересобираются ФИЧИ по состоянию на as_of этого фолда. Если фичи посчитать
один раз на всей истории и потом нарезать — получим утечку, и никакой
gap уже не спасёт.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from m5.config import Config
from m5.evaluation.cv import Fold
from m5.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FoldResult:
    """Результат одного фолда."""

    fold: Fold
    metrics: dict[str, float]
    n_train_rows: int
    n_val_rows: int
    predictions: pd.DataFrame | None = None
    feature_importance: pd.DataFrame | None = None
    train_seconds: float = 0.0


@dataclass
class BacktestResult:
    """Результат бэктеста целиком."""

    folds: list[FoldResult] = field(default_factory=list)
    baseline_metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        """Среднее и std по фолдам.

        Одно число без разброса — бесполезно. Если WRMSSE скачет
        от 0.55 до 0.85 между фолдами, модель нестабильна, и знать это надо.

        TODO: вернуть {'wrmsse_mean':..., 'wrmsse_std':..., ...}
        """
        raise NotImplementedError

    def beats_baseline(self, metric: str = "wrmsse") -> bool:
        """Побили ли baseline. Это gate для регистрации модели.

        TODO: сравнить summary()[f'{metric}_mean'] с baseline_metrics[metric].
        """
        raise NotImplementedError

    def to_frame(self) -> pd.DataFrame:
        """Таблица «фолд × метрика» — в лог и в MLflow-артефакт."""
        raise NotImplementedError


def run_backtest(
    cfg: Config,
    folds: list[Fold] | None = None,
    log_to_mlflow: bool = True,
    save_predictions: bool = False,
) -> BacktestResult:
    """Полный бэктест.

    Для каждого фолда:
        1. assert_fold_integrity(fold, horizon)
        2. build_features(cfg, as_of=fold.train_end)   <- фичи пересобираются!
        3. train_model на train-отрезке
        4. predict на val-отрезке
        5. metrics_report(факт, прогноз, weights)
        6. лог в MLflow как вложенный run

    TODO: реализовать цикл, поднять MLflow parent run, собрать BacktestResult.
    """
    raise NotImplementedError


def compare_models(results: dict[str, BacktestResult], metric: str = "wrmsse") -> pd.DataFrame:
    """Сравнить несколько конфигураций на одних и тех же фолдах.

    Основной сценарий — ablation по внешним данным:
        baseline_features        — только M5
        + weather                — плюс Open-Meteo
        + holidays               — плюс Nager.Date
        + weather + holidays     — всё вместе

    Без такого сравнения нельзя утверждать, что внешние данные вообще помогли.
    Результат этой таблицы идёт прямо в README.

    TODO: собрать DataFrame (конфигурация × фолд × метрика) + дельта к базовой.
    """
    raise NotImplementedError


def ablation_external_data(cfg: Config) -> pd.DataFrame:
    """Прогнать ablation по внешним источникам.

    TODO: для каждой комбинации (weather on/off, holidays on/off)
          сделать override конфига, вызвать run_backtest, сравнить.
    """
    raise NotImplementedError


def horizon_breakdown(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Метрики в разбивке по горизонту 1..28.

    Качество обязано деградировать с ростом горизонта. Если не деградирует —
    почти наверняка где-то утечка, и это первое, что надо проверять.

    TODO: сгруппировать по horizon, посчитать wmape/mae/bias.
    """
    raise NotImplementedError


def error_analysis(predictions: pd.DataFrame, actuals: pd.DataFrame) -> dict[str, Any]:
    """Где именно модель ошибается: топ худших рядов, срезы по категориям.

    Понимать структуру ошибки важнее, чем знать её среднее.

    TODO: топ-50 худших рядов по вкладу в WRMSSE, срезы по cat_id/store_id,
          отдельно — прерывистые ряды (zero_ratio > 0.8).
    """
    raise NotImplementedError
