"""Бэктест: прогнать модель по всем фолдам и собрать честные метрики.

Отличие от «посчитал скор на валидации»: здесь модель переобучается на каждом
фолде заново, а метрики собираются со всех четырёх. Одно число с одного фолда
не показывает главного — стабильна модель или её качество скачет.

ПРО ПЕРЕСБОРКУ ФИЧ НА КАЖДОМ ФОЛДЕ
    Классическое требование — пересобирать фичи по состоянию на as_of каждого
    фолда, иначе будет утечка. Здесь витрина собирается один раз, а фолды
    нарезаются срезом по датам, и это ЭКВИВАЛЕНТНО пересборке.

    Почему: каждая фича строки за дату T зависит только от данных <= T - 28
    (все лаги и окна сдвинуты на горизонт). Значение фичи не меняется от того,
    какой as_of стоял при сборке витрины — оно определяется самой датой строки.
    Проверяется тестом test_feature_value_stable_when_future_changes.

    Если когда-нибудь появятся фичи, зависящие от as_of (например, «дней
    с последней поставки на момент прогноза»), это рассуждение сломается,
    и пересборку придётся вернуть. Тогда — флаг rebuild_features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from m5.config import Config
from m5.data.access import data_bounds, list_stores, load_mart, load_sales
from m5.evaluation.cv import Fold, assert_fold_integrity, make_folds
from m5.utils.logging import get_logger

logger = get_logger(__name__)

METRIC_COLUMNS = ["wmape", "mae", "rmse", "bias"]


@dataclass
class FoldResult:
    """Результат одного фолда."""

    fold: Fold
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    n_train_rows: int
    n_val_rows: int
    predictions: pd.DataFrame | None = None
    feature_importance: pd.DataFrame | None = None


@dataclass
class BacktestResult:
    """Результат бэктеста целиком."""

    folds: list[FoldResult] = field(default_factory=list)
    baseline_metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        """Среднее и std по фолдам.

        std здесь не для красоты: если WMAPE скачет от 0.45 до 0.62 между
        фолдами, модель нестабильна, и среднее 0.53 вводит в заблуждение.
        """
        frame = self.to_frame()
        out: dict[str, float] = {}
        for column in METRIC_COLUMNS:
            if column in frame:
                out[f"{column}_mean"] = float(frame[column].mean())
                out[f"{column}_std"] = float(frame[column].std(ddof=0))
        out["n_folds"] = float(len(self.folds))
        return out

    def beats_baseline(self, metric: str = "wmape", min_improvement: float = 0.0) -> bool:
        """Побили ли baseline. Это gate для регистрации модели."""
        model_score = self.summary().get(f"{metric}_mean")
        baseline_score = self.baseline_metrics.get(metric)
        if model_score is None or baseline_score is None:
            return False
        return model_score < baseline_score * (1 - min_improvement)

    def to_frame(self) -> pd.DataFrame:
        """Таблица «фолд × метрика» — в лог и в артефакт MLflow."""
        rows = []
        for item in self.folds:
            row = {
                "fold": item.fold.index,
                "train_end": item.fold.train_end,
                "val_start": item.fold.val_start,
                "n_train": item.n_train_rows,
                "n_val": item.n_val_rows,
            }
            row.update({k: v for k, v in item.metrics.items() if k in METRIC_COLUMNS})
            row.update(
                {
                    f"baseline_{k}": v
                    for k, v in item.baseline_metrics.items()
                    if k in METRIC_COLUMNS
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)


def run_backtest(
    cfg: Config,
    folds: list[Fold] | None = None,
    log_to_mlflow: bool = True,
    save_predictions: bool = False,
) -> BacktestResult:
    """Полный бэктест по всем фолдам rolling origin."""
    from m5.models.baseline import dow_mean, score
    from m5.models.train import feature_importance, predict_frame, train_one

    horizon = int(cfg.project.horizon)
    data_start, data_end = data_bounds(cfg)
    folds = folds or make_folds(cfg, data_start, data_end)
    stores = list_stores(cfg) if cfg.model.get("split_by") == "store_id" else [None]

    result = BacktestResult()

    for fold in folds:
        assert_fold_integrity(fold, horizon)
        logger.info("Бэктест %s", fold.describe())

        models = {}
        predictions = []
        n_train = 0

        for store_id in stores:
            train_df = load_mart(cfg, fold.train_start, fold.train_end, store_id).dropna(
                subset=["sales"]
            )
            valid_df = load_mart(cfg, fold.val_start, fold.val_end, store_id).dropna(
                subset=["sales"]
            )
            if train_df.empty or valid_df.empty:
                logger.warning("fold=%d store=%s: пустая выборка, пропускаю", fold.index, store_id)
                continue

            model = train_one(train_df, valid_df, cfg, store_id=store_id)
            models[model.key] = model
            predictions.append(predict_frame(model, valid_df))
            n_train += len(train_df)

        if not predictions:
            raise RuntimeError(f"Фолд {fold.index}: не обучено ни одной модели")

        preds = pd.concat(predictions, ignore_index=True)
        actual = load_sales(cfg, fold.val_start, fold.val_end)
        metrics = score(actual, preds)

        # Baseline на ТОМ ЖЕ фолде: сравнение имеет смысл только на одних данных
        history = load_sales(
            cfg, fold.forecast_origin - pd.Timedelta(days=90).to_pytimedelta(), fold.forecast_origin
        )
        baseline_preds = dow_mean(history, fold.forecast_origin, horizon=horizon)
        baseline = score(actual, baseline_preds)

        logger.info(
            "fold=%d модель wmape=%.4f | baseline wmape=%.4f | выигрыш %.1f%%",
            fold.index,
            metrics["wmape"],
            baseline["wmape"],
            (1 - metrics["wmape"] / baseline["wmape"]) * 100,
        )

        result.folds.append(
            FoldResult(
                fold=fold,
                metrics=metrics,
                baseline_metrics=baseline,
                n_train_rows=n_train,
                n_val_rows=len(actual),
                predictions=preds if save_predictions else None,
                feature_importance=feature_importance(models, top_n=30),
            )
        )

    result.baseline_metrics = {
        column: float(pd.Series([f.baseline_metrics[column] for f in result.folds]).mean())
        for column in METRIC_COLUMNS
    }

    summary = result.summary()
    logger.info(
        "Итог бэктеста: wmape %.4f ± %.4f по %d фолдам | baseline %.4f | побили: %s",
        summary["wmape_mean"],
        summary["wmape_std"],
        len(result.folds),
        result.baseline_metrics["wmape"],
        result.beats_baseline(),
    )

    if log_to_mlflow:
        try:
            from m5.models.registry import log_backtest

            log_backtest(result, cfg)
        except Exception as exc:
            logger.warning("MLflow недоступен (%s) — бэктест не залогирован", exc)

    return result


def compare_models(results: dict[str, BacktestResult], metric: str = "wmape") -> pd.DataFrame:
    """Сравнить несколько конфигураций на одних и тех же фолдах.

    Основной сценарий — ablation по внешним данным: без них / +погода /
    +праздники / всё вместе. Без такого сравнения нельзя утверждать,
    что внешние источники вообще помогли.
    """
    rows = []
    for name, result in results.items():
        summary = result.summary()
        rows.append(
            {
                "конфигурация": name,
                f"{metric}_mean": summary.get(f"{metric}_mean"),
                f"{metric}_std": summary.get(f"{metric}_std"),
                "побил baseline": result.beats_baseline(metric),
            }
        )
    frame = pd.DataFrame(rows)

    if len(frame) > 1:
        base = frame[f"{metric}_mean"].iloc[0]
        frame["дельта к базовой, %"] = ((frame[f"{metric}_mean"] / base - 1) * 100).round(2)
    return frame


def ablation_external_data(cfg: Config) -> pd.DataFrame:
    """Прогнать ablation по внешним источникам.

    TODO: включается после реализации m5.external — сейчас погодные и
    праздничные фичи всегда NULL, и сравнивать нечего.
    """
    raise NotImplementedError(
        "Ablation имеет смысл после реализации external/: сейчас погода и праздники — заглушки"
    )


def horizon_breakdown(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    """Метрики в разбивке по горизонту 1..28.

    Качество обязано деградировать с ростом горизонта. Если не деградирует —
    почти наверняка утечка, и это первое, что надо проверять.
    """
    from m5.evaluation.metrics import bias, mae, wmape

    merged = actuals.merge(predictions, on=["id", "date"], how="inner")
    if "horizon" not in merged:
        raise ValueError("В прогнозе нет колонки horizon")

    rows = []
    for horizon, group in merged.groupby("horizon"):
        y_true = group["sales"].to_numpy(dtype=float)
        y_pred = group["prediction"].to_numpy(dtype=float)
        rows.append(
            {
                "horizon": int(horizon),
                "wmape": wmape(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "bias": bias(y_true, y_pred),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values("horizon")


def error_analysis(
    predictions: pd.DataFrame, actuals: pd.DataFrame, top_n: int = 50
) -> dict[str, Any]:
    """Где именно модель ошибается: топ худших рядов и срезы.

    Понимать структуру ошибки важнее, чем знать её среднее: закупщику
    интересны конкретные позиции, по которым он регулярно промахивается.
    """
    merged = actuals.merge(predictions, on=["id", "date"], how="inner")
    merged["abs_error"] = (merged["sales"] - merged["prediction"]).abs()

    by_series = (
        merged.groupby("id")
        .agg(abs_error=("abs_error", "sum"), sales=("sales", "sum"))
        .sort_values("abs_error", ascending=False)
    )
    by_series["доля общей ошибки, %"] = (
        by_series["abs_error"] / by_series["abs_error"].sum() * 100
    ).round(2)

    return {
        "худшие ряды": by_series.head(top_n),
        "всего ошибки": float(merged["abs_error"].sum()),
        "доля топ-50 в ошибке, %": float(by_series.head(top_n)["доля общей ошибки, %"].sum()),
    }
