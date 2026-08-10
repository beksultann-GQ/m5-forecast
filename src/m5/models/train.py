"""Обучение LightGBM.

Решения и почему они такие:

objective=tweedie, variance_power=1.1
    ~68% таргета — нули, распределение сильно скошено. Tweedie — это смесь
    пуассоновского процесса (сколько покупок) и гамма-распределения (размер покупки),
    ровно та структура, что у прерывистого розничного спроса.
    На M5 tweedie стабильно бьёт rmse и poisson.

split_by=store_id
    10 моделей вместо одной. Меньше пик памяти (влезает в ноутбук),
    и модель магазина ловит его специфику. Обучаются независимо.

ПОЧЕМУ ГРУППЫ ГОРИЗОНТОВ СЕЙЧАС НЕ ИСПОЛЬЗУЮТСЯ
    В conf/model.yaml описаны horizon_groups (1-7, 8-14, 15-21, 22-28) —
    классический direct multi-step. Но при текущем наборе фич они бессмысленны,
    и это надо понимать, а не тащить «потому что в плане было написано».

    Все фичи построены относительно даты строки T и используют лаги >= 28.
    Для прогноза дня as_of + h (h = 1..28) фича lag_28 берёт день as_of + h - 28,
    то есть НЕ ПОЗЖЕ as_of при любом h. Значит один и тот же вектор фич валиден
    для всего горизонта, и четыре модели на группах обучались бы на идентичных
    данных — это просто вчетверо больше вычислений при том же результате.

    Direct multi-step даёт выигрыш тогда, когда у ближних горизонтов фичи СВЕЖЕЕ:
    для h = 1..7 законно использовать lag_7, для h = 8..14 — lag_14 и так далее.
    Это следующий шаг: параметризовать минимальный лаг в SQL фич и собирать
    четыре витрины. Пока — одна модель на магазин на весь горизонт.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import lightgbm as lgb
import pandas as pd

from m5.config import Config
from m5.data.access import data_bounds, list_stores, load_mart, load_sales
from m5.utils.logging import get_logger
from m5.utils.paths import ensure_dir, resolve

logger = get_logger(__name__)


@dataclass
class TrainedModel:
    """Обученная модель + всё, без чего её нельзя честно применить.

    feature_names и categories тут не для красоты: если на инференсе
    порядок колонок или коды категорий разъедутся, LightGBM не упадёт —
    он молча предскажет ерунду. Поэтому сохраняем и сверяем.
    """

    booster: lgb.Booster
    feature_names: list[str]
    categorical_features: list[str]
    categories: dict[str, list[str]]
    store_id: str | None
    params: dict[str, Any]
    best_iteration: int
    train_end: date
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.store_id or 'all'}"

    def save(self, directory: str | Path) -> Path:
        """Сохранить booster + метаданные рядом."""
        target = ensure_dir(directory) / self.key
        target.mkdir(parents=True, exist_ok=True)

        self.booster.save_model(str(target / "model.txt"), num_iteration=self.best_iteration)
        meta = {
            "feature_names": self.feature_names,
            "categorical_features": self.categorical_features,
            "categories": self.categories,
            "store_id": self.store_id,
            "params": self.params,
            "best_iteration": self.best_iteration,
            "train_end": self.train_end.isoformat(),
            "metrics": self.metrics,
        }
        (target / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, directory: str | Path, key: str) -> TrainedModel:
        source = resolve(directory) / key
        meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
        return cls(
            booster=lgb.Booster(model_file=str(source / "model.txt")),
            feature_names=meta["feature_names"],
            categorical_features=meta["categorical_features"],
            categories=meta["categories"],
            store_id=meta["store_id"],
            params=meta["params"],
            best_iteration=meta["best_iteration"],
            train_end=date.fromisoformat(meta["train_end"]),
            metrics=meta.get("metrics", {}),
        )


class EmptyValidationError(RuntimeError):
    """Валидационное окно пустое — обучаться вслепую нельзя."""


def clamp_as_of(as_of: date | str | None, data_max: date) -> date:
    """Ограничить as_of последней датой, для которой есть факт.

    Зачем это нужно: Airflow подставляет в таск `--as-of {{ ds }}` — календарную
    дату запуска. В реальном проде это правильно, но на историческом датасете
    (M5 заканчивается 2016-06-19) `ds` уезжает в будущее. Тогда rolling origin
    нарезает валидацию на датах, где данных нет, valid_df оказывается пустым,
    early stopping не работает, а метрики не считаются — и всё это молча.

    Поэтому обрезаем и громко предупреждаем.
    """
    if as_of is None:
        return data_max

    requested = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
    if requested > data_max:
        logger.warning(
            "as_of=%s позже последней даты с фактом (%s) — обрезаю до неё. "
            "Если это прод, значит данные не доехали.",
            requested,
            data_max,
        )
        return data_max
    return requested


# ---------------------------------------------------------------- подготовка фич


NON_FEATURE_COLUMNS = {
    "id",
    "date",
    "sales",
    "weather_source",
    "days_since_first_sale",
    "store_id",  # при обучении по магазинам колонка константна
}


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Колонки-фичи в детерминированном порядке.

    Порядок фиксируем один раз и сохраняем в модель: LightGBM работает
    с позициями, и разъехавшийся порядок не вызовет ошибку — просто
    предсказания станут мусором.
    """
    return [col for col in frame.columns if col not in NON_FEATURE_COLUMNS]


def prepare_dataset(
    frame: pd.DataFrame,
    feature_names: list[str],
    categories: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, list[str]]]:
    """DataFrame -> (X, y, categories).

    Args:
        categories: если передан — используем ЭТИ категории (валидация/инференс),
            иначе фиксируем по обучающей выборке и возвращаем.
    """
    features = frame[feature_names].copy()
    resolved: dict[str, list[str]] = {}

    for col in features.columns:
        if features[col].dtype == object or isinstance(features[col].dtype, pd.CategoricalDtype):
            known = categories[col] if categories and col in categories else None
            if known is None:
                known = sorted(features[col].dropna().astype(str).unique().tolist())
            features[col] = pd.Categorical(features[col].astype("string"), categories=known)
            resolved[col] = known

    target = frame["sales"].astype(float)
    return features, target, resolved


def categorical_feature_names(categories: dict[str, list[str]]) -> list[str]:
    return sorted(categories)


# ---------------------------------------------------------------- обучение


def train_one(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None,
    cfg: Config,
    store_id: str | None = None,
) -> TrainedModel:
    """Обучить одну модель (один магазин, весь горизонт)."""
    started = time.perf_counter()

    feature_names = feature_columns(train_df)
    x_train, y_train, categories = prepare_dataset(train_df, feature_names)
    cat_features = categorical_feature_names(categories)

    train_set = lgb.Dataset(
        x_train, label=y_train, categorical_feature=cat_features, free_raw_data=True
    )

    valid_sets = [train_set]
    valid_names = ["train"]
    if valid_df is not None and len(valid_df):
        x_valid, y_valid, _ = prepare_dataset(valid_df, feature_names, categories)
        valid_sets.append(
            lgb.Dataset(
                x_valid,
                label=y_valid,
                categorical_feature=cat_features,
                reference=train_set,
                free_raw_data=True,
            )
        )
        valid_names.append("valid")

    params = dict(cfg.model.params)
    training = cfg.model.training

    callbacks = [lgb.log_evaluation(period=int(training.log_evaluation_period))]
    if len(valid_sets) > 1:
        callbacks.append(lgb.early_stopping(int(training.early_stopping_rounds), verbose=False))

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=int(training.num_boost_round),
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    elapsed = time.perf_counter() - started
    logger.info(
        "store=%s обучена за %.1f с, итераций %d, строк %s",
        store_id or "all",
        elapsed,
        booster.best_iteration or booster.current_iteration(),
        f"{len(train_df):,}",
    )

    return TrainedModel(
        booster=booster,
        feature_names=feature_names,
        categorical_features=cat_features,
        categories=categories,
        store_id=store_id,
        params=params,
        best_iteration=booster.best_iteration or booster.current_iteration(),
        train_end=train_df["date"].max().date(),
        metrics={"train_seconds": round(elapsed, 1), "n_train_rows": len(train_df)},
    )


def train(
    cfg: Config,
    as_of: date | str | None = None,
    log_to_mlflow: bool = True,
    stores: list[str] | None = None,
) -> dict[str, TrainedModel]:
    """Обучить ансамбль: по одной модели на магазин.

    Валидация — фолд 0 из rolling origin (самый свежий отрезок).
    """
    from m5.evaluation.cv import make_folds
    from m5.models.baseline import score

    data_start, data_max = data_bounds(cfg)
    data_end = clamp_as_of(as_of, data_max)

    fold = make_folds(cfg, data_start, data_end)[0]
    logger.info("Обучение на фолде: %s", fold.describe())

    target_stores = stores or list_stores(cfg)
    if cfg.model.get("split_by") != "store_id":
        target_stores = [None]

    models: dict[str, TrainedModel] = {}
    predictions: list[pd.DataFrame] = []

    for store_id in target_stores:
        # dropna по таргету: в витрине могут лежать строки будущих дат
        # (их кладёт inference-режим), обучаться на них нечему
        train_df = load_mart(cfg, fold.train_start, fold.train_end, store_id).dropna(
            subset=["sales"]
        )
        valid_df = load_mart(cfg, fold.val_start, fold.val_end, store_id).dropna(subset=["sales"])

        if train_df.empty:
            logger.warning("store=%s: пустая обучающая выборка, пропускаю", store_id)
            continue

        model = train_one(train_df, valid_df, cfg, store_id=store_id)
        models[model.key] = model

        if not valid_df.empty:
            predictions.append(predict_frame(model, valid_df))

    if not models:
        raise RuntimeError("Не обучено ни одной модели — проверь, что ft.mart непустая")

    if not predictions:
        # Молчаливо обучиться без валидации — худший исход: early stopping
        # не работает, метрик нет, а модель выглядит нормальной и едет дальше.
        raise EmptyValidationError(
            f"Валидационное окно [{fold.val_start}..{fold.val_end}] пустое — "
            f"модель обучилась бы без early stopping и без метрик.\n"
            f"Обычно это значит, что as_of указывает за пределы данных. "
            f'Проверь `m5 db sql "SELECT MAX(date) FROM ft.mart"`.'
        )

    metrics: dict[str, float] = {}
    if predictions:
        preds = pd.concat(predictions, ignore_index=True)
        actual = load_sales(cfg, fold.val_start, fold.val_end)
        metrics = score(actual, preds)
        logger.info(
            "Валидация фолда %d: wmape=%.4f mae=%.4f rmse=%.4f bias=%+.4f",
            fold.index,
            metrics["wmape"],
            metrics["mae"],
            metrics["rmse"],
            metrics["bias"],
        )

    model_dir = _save_ensemble(cfg, models, metrics, fold)
    logger.info("Ансамбль сохранён в %s", model_dir)

    if log_to_mlflow:
        _log_to_mlflow(cfg, models, metrics, fold, model_dir)

    return models


def predict_frame(model: TrainedModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Предсказать на готовом куске витрины. Возвращает (id, date, prediction)."""
    check_feature_consistency(model, frame)
    features, _, _ = prepare_dataset(frame, model.feature_names, model.categories)
    raw = model.booster.predict(features, num_iteration=model.best_iteration)

    return pd.DataFrame(
        {
            "id": frame["id"].to_numpy(),
            "date": frame["date"].to_numpy(),
            # отрицательных продаж не бывает; tweedie обычно не даёт, но проверяем
            "prediction": raw.clip(min=0),
        }
    )


def check_feature_consistency(model: TrainedModel, frame: pd.DataFrame) -> None:
    """Сверить фичи инференса с теми, на которых училась модель.

    Ловит training/serving skew до того, как он доедет до бизнеса.
    """
    missing = [col for col in model.feature_names if col not in frame.columns]
    if missing:
        raise ValueError(f"В данных нет фич, на которых училась модель: {missing[:10]}")


def feature_importance(models: dict[str, TrainedModel], top_n: int | None = None) -> pd.DataFrame:
    """Сводная важность фич по всему ансамблю.

    Нужна не для красоты: именно здесь видно, попали ли погода и праздники
    в топ или болтаются в хвосте. Если внешние фичи внизу и ablation
    не показывает прироста — их надо убрать, а не оставлять «для солидности».
    """
    frames = []
    for key, model in models.items():
        gain = model.booster.feature_importance(importance_type="gain")
        split = model.booster.feature_importance(importance_type="split")
        frames.append(
            pd.DataFrame(
                {
                    "model": key,
                    "feature": model.booster.feature_name(),
                    "gain": gain,
                    "split": split,
                }
            )
        )

    combined = pd.concat(frames, ignore_index=True)
    summary = (
        combined.groupby("feature")[["gain", "split"]]
        .mean()
        .sort_values("gain", ascending=False)
        .reset_index()
    )
    summary["gain_share"] = summary["gain"] / summary["gain"].sum()
    return summary.head(top_n) if top_n else summary


# ---------------------------------------------------------------- сохранение


def _save_ensemble(cfg: Config, models: dict[str, TrainedModel], metrics, fold) -> Path:
    """Сохранить модели, важность фич и сводку прогона."""
    root = ensure_dir(Path(cfg.paths.processed_dir) / "models" / "latest")
    for model in models.values():
        model.save(root)

    feature_importance(models).to_csv(root / "feature_importance.csv", index=False)
    (root / "run.json").write_text(
        json.dumps(
            {
                "fold": fold.describe(),
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "val_start": fold.val_start.isoformat(),
                "val_end": fold.val_end.isoformat(),
                "forecast_origin": fold.forecast_origin.isoformat(),
                "n_models": len(models),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def _log_to_mlflow(cfg: Config, models, metrics, fold, model_dir: Path) -> None:
    """Залогировать прогон и завести версию модели в Registry.

    Падение MLflow не должно ронять обучение: модель уже сохранена локально,
    терять её из-за недоступного трекинг-сервера бессмысленно.
    """
    try:
        import mlflow

        from m5.config.loader import config_hash, flatten, git_sha
        from m5.data.download import dataset_version
        from m5.models.registry import log_model_ensemble, save_run_id, setup_mlflow

        setup_mlflow(cfg)

        with mlflow.start_run(run_name=f"lgbm_tweedie_{fold.train_end}") as run:
            mlflow.set_tags(
                {
                    "git_sha": git_sha(),
                    "config_hash": config_hash(cfg),
                    "dataset_version": dataset_version(Path(cfg.paths.raw_dir)),
                    "fold": fold.describe(),
                    "n_models": len(models),
                }
            )
            mlflow.log_params({k: v for k, v in flatten(cfg).items() if v is not None})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})

            # Ансамбль уезжает в Registry сразу в Staging. Промоушен в Production —
            # отдельное решение по метрикам, см. registry.promote_if_better.
            log_model_ensemble(cfg, model_dir, register=True)
            save_run_id(model_dir, run.info.run_id)
            logger.info("MLflow run %s залогирован, модель зарегистрирована", run.info.run_id)
    except Exception as exc:
        logger.warning(
            "MLflow недоступен (%s) — прогон не залогирован, модель сохранена локально", exc
        )
