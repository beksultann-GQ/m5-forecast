"""MLflow: трекинг экспериментов и Model Registry.

Именно этот слой отличает «я обучил модель» от «я умею вести ML в проде».
Через полгода надо мочь ответить: какая модель сейчас в проде, на каких данных
она обучена, каким коммитом, какие метрики показала на бэктесте.

Ансамбль из 10 моделей по магазинам оборачивается в один pyfunc: снаружи это
одна модель с одним predict(), и Registry версионирует её как единое целое.
Иначе «версия модели» распалась бы на десять несогласованных кусков.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from m5.config import Config
from m5.utils.logging import get_logger
from m5.utils.paths import resolve

logger = get_logger(__name__)

STAGE_NONE = "None"
STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"

ARTIFACT_PATH = "model"
ENSEMBLE_ARTIFACT = "ensemble"


def setup_mlflow(cfg: Config) -> str:
    """Настроить tracking URI и эксперимент. Возвращает experiment_id."""
    import mlflow

    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    experiment = mlflow.set_experiment(str(cfg.mlflow.experiment_name))
    return experiment.experiment_id


def _make_pyfunc_model():
    """Собрать pyfunc-обёртку над ансамблем «модель на магазин».

    Класс определён внутри функции, а не на уровне модуля, чтобы импорт mlflow
    был отложенным: этот модуль подтягивается из predict.py, а mlflow нужен
    не в каждом сценарии (например, инференсу с локальной моделью — нет).
    """
    import mlflow

    class _Ensemble(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from m5.models.train import TrainedModel

            root = Path(context.artifacts[ENSEMBLE_ARTIFACT])
            self.models = {
                p.name: TrainedModel.load(root, p.name)
                for p in sorted(root.iterdir())
                if p.is_dir()
            }

        def predict(self, context, model_input, params=None):
            """Маршрутизируем строки по магазинам к своим бустерам."""
            from m5.models.train import predict_frame

            frames = []
            for model in self.models.values():
                rows = (
                    model_input[model_input["store_id"] == model.store_id]
                    if model.store_id
                    else model_input
                )
                if len(rows):
                    frames.append(predict_frame(model, rows))
            if not frames:
                return pd.DataFrame(columns=["id", "date", "prediction"])
            return pd.concat(frames, ignore_index=True)

    return _Ensemble()


def log_model_ensemble(cfg: Config, model_dir: Path, register: bool = True) -> str:
    """Залогировать ансамбль как pyfunc и (опционально) завести версию в Registry.

    Новая версия всегда приезжает в Staging, а не сразу в Production:
    промоушен — отдельное решение, принимаемое по метрикам (см. promote_if_better).

    Returns:
        model_uri вида runs:/<run_id>/model
    """
    import mlflow

    info = mlflow.pyfunc.log_model(
        name=ARTIFACT_PATH,
        python_model=_make_pyfunc_model(),
        artifacts={ENSEMBLE_ARTIFACT: str(resolve(model_dir))},
        registered_model_name=str(cfg.mlflow.registered_model_name) if register else None,
    )

    if register:
        version = latest_version(cfg)
        if version is not None:
            transition(cfg, version, STAGE_STAGING)
            logger.info("Зарегистрирована версия %s в стадии %s", version, STAGE_STAGING)

    return info.model_uri


def latest_version(cfg: Config) -> int | None:
    """Номер последней зарегистрированной версии."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=str(cfg.mlflow.tracking_uri))
    name = str(cfg.mlflow.registered_model_name)
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        return None
    return max(int(v.version) for v in versions)


def transition(cfg: Config, version: int, stage: str, archive_existing: bool = False) -> None:
    """Перевести версию в стадию.

    Стадии в MLflow 2.9+ считаются устаревшими в пользу алиасов, но здесь
    оставлены сознательно: Staging/Production — общепринятый словарь,
    и на собеседовании про него спросят именно в этих терминах.
    """
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=str(cfg.mlflow.tracking_uri))
    client.transition_model_version_stage(
        name=str(cfg.mlflow.registered_model_name),
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing,
    )


def current_production_model(cfg: Config) -> dict[str, Any] | None:
    """Метаданные текущей прод-модели: версия, run_id, метрики."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=str(cfg.mlflow.tracking_uri))
    name = str(cfg.mlflow.registered_model_name)

    versions = [
        v
        for v in client.search_model_versions(f"name='{name}'")
        if v.current_stage == STAGE_PRODUCTION
    ]
    if not versions:
        return None

    newest = max(versions, key=lambda v: int(v.version))
    run = client.get_run(newest.run_id)
    return {
        "version": int(newest.version),
        "run_id": newest.run_id,
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
    }


def compare_with_production(
    cfg: Config,
    candidate_version: int | None = None,
    metric: str = "wmape",
    min_improvement: float = 0.01,
) -> tuple[bool, str, int | None]:
    """Сравнить кандидата с прод-моделью. Ничего не меняет — только решает.

    Вынесено отдельно от промоушена, потому что решение и действие нужны
    в разных местах: DAG сначала спрашивает «лучше ли» в ShortCircuitOperator,
    и только потом отдельным таском промоутит.

    Returns:
        (лучше_ли, причина_текстом, версия_кандидата)

    Причина текстом обязательна: она уезжает в XCom и в алерт, и дежурный
    должен понять решение, не открывая код.
    """
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=str(cfg.mlflow.tracking_uri))
    candidate_version = candidate_version or latest_version(cfg)
    if candidate_version is None:
        return False, "В Registry нет ни одной версии — нечего промоутить", None

    candidate = client.get_model_version(
        str(cfg.mlflow.registered_model_name), str(candidate_version)
    )
    candidate_score = dict(client.get_run(candidate.run_id).data.metrics).get(metric)
    if candidate_score is None:
        return False, f"У версии {candidate_version} нет метрики {metric}", candidate_version

    production = current_production_model(cfg)
    if production is None:
        return (
            True,
            f"Production-модели не было, версия {candidate_version} становится первой "
            f"({metric}={candidate_score:.4f})",
            candidate_version,
        )

    prod_score = production["metrics"].get(metric)
    if prod_score is None:
        return (
            True,
            f"У прод-версии {production['version']} нет метрики {metric} — заменяем",
            candidate_version,
        )

    # Для wmape/wrmsse/mae меньше = лучше
    threshold = prod_score * (1 - min_improvement)
    better = candidate_score < threshold
    verdict = "лучше порога, промоутим" if better else "недостаточно лучше, прод остаётся"

    reason = (
        f"кандидат v{candidate_version} {metric}={candidate_score:.4f}, "
        f"прод v{production['version']} {metric}={prod_score:.4f}, "
        f"порог {threshold:.4f} (min_improvement={min_improvement:.0%}) — {verdict}"
    )
    return better, reason, candidate_version


def promote_if_better(
    cfg: Config,
    candidate_version: int | None = None,
    metric: str = "wmape",
    min_improvement: float = 0.01,
) -> bool:
    """Промоутить кандидата в Production, только если он лучше текущего прода.

    Тот самый автоматический гейт: новая модель НЕ едет в прод просто потому,
    что она новая.

    Args:
        min_improvement: минимальное ОТНОСИТЕЛЬНОЕ улучшение (1% по умолчанию).
            Меньший выигрыш — разброс между фолдами, а не прогресс. Без порога
            прод дёргался бы на шуме каждую неделю.
    """
    better, reason, version = compare_with_production(
        cfg, candidate_version, metric, min_improvement
    )
    logger.info("Гейт промоушена: %s", reason)

    if better and version is not None:
        transition(cfg, version, STAGE_PRODUCTION, archive_existing=True)
    return better


def load_registry_models(cfg: Config, stage: str = STAGE_PRODUCTION) -> tuple[dict, str, str]:
    """Загрузить ансамбль из Registry по стадии.

    Returns:
        (модели, model_version, run_id)

    Raises:
        RuntimeError: если в этой стадии нет версий.
    """
    import mlflow
    from mlflow import MlflowClient

    from m5.models.train import TrainedModel

    client = MlflowClient(tracking_uri=str(cfg.mlflow.tracking_uri))
    name = str(cfg.mlflow.registered_model_name)
    versions = [
        v for v in client.search_model_versions(f"name='{name}'") if v.current_stage == stage
    ]
    if not versions:
        raise RuntimeError(f"В Registry нет версий модели {name} в стадии {stage}")

    newest = max(versions, key=lambda v: int(v.version))
    mlflow.set_tracking_uri(str(cfg.mlflow.tracking_uri))
    local = Path(mlflow.artifacts.download_artifacts(artifact_uri=newest.source))

    ensemble_dir = _resolve_ensemble_dir(local)
    models = {
        p.name: TrainedModel.load(ensemble_dir, p.name)
        for p in sorted(ensemble_dir.iterdir())
        if p.is_dir()
    }
    logger.info(
        "Из Registry загружено %d моделей, версия %s (%s)", len(models), newest.version, stage
    )

    return models, f"registry:v{newest.version}", newest.run_id


def _resolve_ensemble_dir(model_root: Path) -> Path:
    """Найти каталог с моделями внутри скачанного pyfunc-артефакта.

    MLflow кладёт переданный каталог под его СОБСТВЕННЫМ именем, а не под
    ключом из `artifacts={...}`: мы передаём .../models/latest, и внутри
    оказывается artifacts/latest, а не artifacts/ensemble. Полагаться на
    имя каталога нельзя, поэтому ищем по содержимому — по наличию meta.json
    у вложенных моделей.
    """
    artifacts = model_root / "artifacts"
    if not artifacts.exists():
        raise RuntimeError(f"В артефактах модели нет каталога artifacts: {model_root}")

    candidates = [
        artifacts / ENSEMBLE_ARTIFACT,
        *sorted(p for p in artifacts.iterdir() if p.is_dir()),
    ]
    for candidate in candidates:
        if candidate.exists() and any(
            (sub / "meta.json").exists() for sub in candidate.iterdir() if sub.is_dir()
        ):
            return candidate

    raise RuntimeError(f"Не нашёл каталог с моделями внутри {artifacts}")


def log_backtest(result, cfg: Config) -> None:
    """Залогировать результаты бэктеста: каждый фолд — вложенный run.

    Так в UI видно и агрегат, и разброс между фолдами. Одно среднее без
    разброса скрывает нестабильную модель.
    """
    import mlflow

    setup_mlflow(cfg)
    summary = result.summary()

    with mlflow.start_run(run_name="backtest"):
        mlflow.log_metrics(summary)
        for fold_result in result.folds:
            with mlflow.start_run(run_name=f"fold_{fold_result.fold.index}", nested=True):
                mlflow.set_tag("fold", fold_result.fold.describe())
                mlflow.log_metrics(
                    {k: v for k, v in fold_result.metrics.items() if isinstance(v, int | float)}
                )
        if result.baseline_metrics:
            mlflow.log_metrics({f"baseline_{k}": v for k, v in result.baseline_metrics.items()})
        mlflow.log_text(result.to_frame().to_string(), "backtest_by_fold.txt")


def save_run_id(model_dir: Path, run_id: str) -> None:
    """Дописать mlflow_run_id в run.json ансамбля.

    Нужно, чтобы по локальной модели можно было найти её прогон в MLflow,
    а не только наоборот.
    """
    path = resolve(model_dir) / "run.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    meta["mlflow_run_id"] = run_id
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
