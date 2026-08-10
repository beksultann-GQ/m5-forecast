"""Инференс: прогноз на 28 дней вперёд в витрину mart.forecast.

Фичи считаются ТЕМ ЖЕ кодом и тем же SQL, что и на обучении —
`build_features(mode='inference')`. Отдельного «скрипта для прода» нет
и не должно быть: именно из расхождения между train- и inference-кодом
рождается training/serving skew.

Разница между режимами ровно одна: витрина тянется на 28 дней дальше,
и у этих строк sales = NULL. Всё остальное идентично.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from m5.config import Config
from m5.data.access import data_bounds, load_mart, load_sales
from m5.data.duck import connect
from m5.utils.logging import get_logger
from m5.utils.paths import resolve

logger = get_logger(__name__)

LOCAL_MODEL_DIR = "models/latest"


class SanityCheckError(RuntimeError):
    """Прогноз не прошёл проверки — в витрину не пишем."""


def predict(
    cfg: Config,
    as_of: date | str | None = None,
    model_stage: str = "Production",
    write_to_mart: bool = True,
) -> pd.DataFrame:
    """Прогноз на horizon дней вперёд от as_of.

    Args:
        as_of: последний день с фактическими продажами. None -> max(date) в данных.
        model_stage: стадия модели в MLflow Registry. Если реестр недоступен,
            откатываемся на локальный ансамбль из data/processed/models/latest.
        write_to_mart: писать ли результат в mart.forecast

    Returns:
        (id, item_id, store_id, date, horizon, prediction, model_version, run_id, as_of)

    Raises:
        SanityCheckError: прогноз не прошёл проверки — в витрину не попадает.
    """
    from m5.models.train import clamp_as_of, predict_frame

    as_of = clamp_as_of(as_of, data_bounds(cfg)[1])
    horizon = int(cfg.project.horizon)
    models, model_version, run_id = load_models(cfg, stage=model_stage)

    logger.info(
        "Инференс: as_of=%s горизонт=%d моделей=%d версия=%s",
        as_of,
        horizon,
        len(models),
        model_version,
    )

    frames: list[pd.DataFrame] = []
    for model in models.values():
        future = load_mart(
            cfg,
            start=as_of + timedelta(days=1),
            end=as_of + timedelta(days=horizon),
            store_id=model.store_id,
        )
        if future.empty:
            logger.warning(
                "store=%s: нет будущих строк в витрине. "
                "Собрана ли витрина в режиме inference? (`m5 features build --mode inference`)",
                model.store_id,
            )
            continue

        preds = predict_frame(model, future)
        preds = preds.merge(
            future[["id", "date", "item_id", "store_id", "sell_price"]],
            on=["id", "date"],
            how="left",
            validate="one_to_one",
        )
        frames.append(preds)

    if not frames:
        raise RuntimeError(
            f"Не получено ни одного прогноза на {as_of}. "
            "Скорее всего витрина собрана в режиме train — в ней нет будущих дат."
        )

    result = pd.concat(frames, ignore_index=True)
    result["horizon"] = (result["date"] - pd.Timestamp(as_of)).dt.days
    result["as_of"] = pd.Timestamp(as_of)
    result["model_version"] = model_version
    result["run_id"] = run_id

    result = post_process(result, cfg)

    checks = sanity_checks(result, cfg, as_of)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SanityCheckError(
            f"Прогноз не прошёл проверки: {failed}. В витрину не записан.\nПолный отчёт: {checks}"
        )
    logger.info("Sanity-checks пройдены: %d из %d", len(checks), len(checks))

    if write_to_mart:
        n_rows = write_forecast(result, cfg)
        logger.info("Записано в mart.forecast: %s строк", f"{n_rows:,}")

    return result


def post_process(preds: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Пост-обработка прогнозов.

    1. clip(0) — отрицательных продаж не бывает. Tweedie обычно не даёт
       отрицательных, но полагаться на «обычно» в проде нельзя.
    2. Зануляем товары вне ассортимента: нет цены на неделю прогноза —
       значит товара не будет на полке, и прогноз спроса бессмысленен.

    Округление до целых НЕ делаем: закупки агрегируют прогноз по позициям,
    и при суммировании дробные значения точнее целых.
    """
    preds = preds.copy()
    preds["prediction"] = preds["prediction"].clip(lower=0)

    out_of_assortment = preds["sell_price"].isna()
    n_zeroed = int(out_of_assortment.sum())
    if n_zeroed:
        preds.loc[out_of_assortment, "prediction"] = 0.0
        logger.info("Занулено %s строк без цены (товара нет в ассортименте)", f"{n_zeroed:,}")

    return preds


def sanity_checks(preds: pd.DataFrame, cfg: Config, as_of: date) -> dict[str, bool]:
    """Проверки перед записью в витрину. Прод-гейт, а не формальность.

    Плохой прогноз хуже отсутствия прогноза: по нему закупят товар.
    Поэтому при любом провале мы НЕ публикуем — fail closed.
    """
    horizon = int(cfg.project.horizon)
    n_series = preds["id"].nunique()

    # Объём прогноза сравниваем с фактом за прошлые horizon дней:
    # ловит катастрофы вроде «подгрузилась не та модель» или «витрина пустая»
    recent = load_sales(cfg, as_of - timedelta(days=horizon - 1), as_of)
    recent_volume = float(recent["sales"].sum()) if not recent.empty else 0.0
    forecast_volume = float(preds["prediction"].sum())
    ratio = forecast_volume / recent_volume if recent_volume else float("inf")

    checks = {
        "нет NaN": not preds["prediction"].isna().any(),
        "нет отрицательных": bool((preds["prediction"] >= 0).all()),
        "нет дублей (id, date)": not preds.duplicated(["id", "date"]).any(),
        "все горизонты 1..N": set(preds["horizon"].unique()) == set(range(1, horizon + 1)),
        "число строк = ряды × горизонт": len(preds) == n_series * horizon,
        "объём в пределах ±40% от факта": 0.6 <= ratio <= 1.4,
    }

    logger.info(
        "Объём прогноза %s против факта за прошлые %d дней %s (отношение %.2f)",
        f"{forecast_volume:,.0f}",
        horizon,
        f"{recent_volume:,.0f}",
        ratio,
    )
    for name, passed in checks.items():
        logger.info("  %s %s", "✓" if passed else "✗", name)

    return checks


def write_forecast(preds: pd.DataFrame, cfg: Config) -> int:
    """Записать прогноз в mart.forecast ИДЕМПОТЕНТНО.

    Идемпотентность обязательна: Airflow ретраит таски, и повторный запуск
    не должен задваивать строки. Реализовано как DELETE по as_of + INSERT
    в одной транзакции — при падении посередине витрина остаётся консистентной.
    """
    as_of = preds["as_of"].iloc[0]
    payload = preds[
        [
            "id",
            "item_id",
            "store_id",
            "date",
            "horizon",
            "prediction",
            "model_version",
            "run_id",
            "as_of",
        ]
    ].copy()
    payload["created_at"] = datetime.now(tz=UTC).replace(tzinfo=None)

    with connect(cfg.paths.duckdb_path) as con:
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM mart.forecast WHERE as_of = ?", [as_of])
            con.register("payload", payload)
            con.execute("INSERT INTO mart.forecast SELECT * FROM payload")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise

    return len(payload)


def load_models(cfg: Config, stage: str = "Production") -> tuple[dict, str, str]:
    """Загрузить ансамбль: сначала пробуем MLflow Registry, потом локальный каталог.

    Возвращаем ещё и версию модели с run_id — они уезжают в витрину рядом
    с каждым прогнозом. Без этого через месяц не ответить, какая модель
    сгенерировала конкретное число, по которому закупили товар.

    Returns:
        (модели, model_version, run_id)
    """
    from m5.models.registry import load_registry_models

    try:
        return load_registry_models(cfg, stage=stage)
    except Exception as exc:
        logger.warning("Registry недоступен (%s) — беру локальный ансамбль", exc)

    return load_local_models(cfg)


def load_local_models(cfg: Config) -> tuple[dict, str, str]:
    """Ансамбль из data/processed/models/latest."""
    from m5.models.train import TrainedModel

    root = resolve(Path(cfg.paths.processed_dir) / LOCAL_MODEL_DIR)
    if not root.exists():
        raise FileNotFoundError(f"Нет обученных моделей в {root}. Сначала `make train`")

    keys = [p.name for p in sorted(root.iterdir()) if p.is_dir()]
    if not keys:
        raise FileNotFoundError(f"В {root} нет моделей")

    models = {key: TrainedModel.load(root, key) for key in keys}

    run_meta = root / "run.json"
    meta = json.loads(run_meta.read_text(encoding="utf-8")) if run_meta.exists() else {}
    version = f"local:{meta.get('train_end', 'unknown')}"

    return models, version, meta.get("mlflow_run_id", "local")


def make_submission(preds: pd.DataFrame, cfg: Config, out_path: str | Path) -> Path:
    """Собрать submission.csv для Kaggle (F1..F28 по каждому id).

    Нужно ровно для одного: сверить свой скор с публичным лидербордом
    и убедиться, что метрика посчитана правильно. В проде не используется.
    """
    wide = preds.pivot_table(index="id", columns="horizon", values="prediction", aggfunc="first")
    wide.columns = [f"F{int(h)}" for h in wide.columns]
    wide = wide.reset_index()

    target = resolve(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(target, index=False)
    logger.info("Submission: %s строк -> %s", f"{len(wide):,}", target)
    return target
