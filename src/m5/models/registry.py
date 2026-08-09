"""MLflow: трекинг экспериментов и Model Registry.

Именно этот слой отличает «я обучил модель» от «я умею вести ML в проде».
Через полгода надо мочь ответить: какая модель сейчас в проде, на каких данных
она обучена, каким коммитом, какие метрики показала на бэктесте.

Логируем всегда:
    параметры       — весь конфиг, плоско
    метрики         — по каждому фолду отдельно + агрегаты
    версия данных   — хеш манифеста data/raw
    git sha         — коммит, которым обучали
    артефакты       — модель, feature_importance, конфиг, таблица бэктеста
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

from m5.config import Config
from m5.utils.logging import get_logger

logger = get_logger(__name__)

STAGE_NONE = "None"
STAGE_STAGING = "Staging"
STAGE_PRODUCTION = "Production"
STAGE_ARCHIVED = "Archived"


def setup_mlflow(cfg: Config) -> str:
    """Настроить tracking URI и эксперимент.

    Returns:
        experiment_id

    TODO: mlflow.set_tracking_uri / set_experiment из cfg.mlflow.
    """
    raise NotImplementedError


@contextmanager
def start_run(cfg: Config, run_name: str, nested: bool = False):
    """Контекст прогона: автоматически проставляет теги воспроизводимости.

    Теги: git_sha, config_hash, dataset_version, host, python_version.

    TODO: обернуть mlflow.start_run, проставить теги, залогировать flatten(cfg).
    """
    raise NotImplementedError


def log_backtest(result, cfg: Config) -> None:
    """Залогировать результаты бэктеста.

    Каждый фолд — вложенный run со своими метриками, родитель — агрегаты.
    Так в UI видно и среднее, и разброс.

    TODO: реализовать.
    """
    raise NotImplementedError


def log_model_ensemble(models: dict, cfg: Config, artifact_path: str = "model") -> str:
    """Залогировать ансамбль (магазины × горизонты) как один артефакт.

    Registry не умеет «40 моделей = одна версия», поэтому оборачиваем ансамбль
    в mlflow.pyfunc.PythonModel: снаружи это одна модель с одним predict().

    Returns:
        model_uri

    TODO: реализовать M5Ensemble(mlflow.pyfunc.PythonModel) и залогировать.
    """
    raise NotImplementedError


def register_model(model_uri: str, cfg: Config, stage: str = STAGE_STAGING) -> int:
    """Зарегистрировать версию модели.

    Returns:
        Номер версии.

    TODO: mlflow.register_model + transition_model_version_stage.
    """
    raise NotImplementedError


def promote_if_better(
    cfg: Config,
    candidate_version: int,
    metric: str = "wrmsse",
    min_improvement: float = 0.01,
) -> bool:
    """Промоутить кандидата в Production, только если он лучше текущего прода.

    Это тот самый автоматический гейт из DAG'а обучения:
    новая модель НЕ едет в прод просто потому, что она новая.

    Args:
        min_improvement: минимальное относительное улучшение (1% по умолчанию).
            Меньший выигрыш — это шум между фолдами, а не прогресс.

    Returns:
        True, если промоутнули.

    TODO:
        1. взять метрику текущей Production-версии
        2. если прода нет — промоутить кандидата
        3. сравнить с учётом min_improvement (для WRMSSE меньше = лучше)
        4. старую версию перевести в Archived
        5. записать решение в лог и в тег версии
    """
    raise NotImplementedError


def current_production_model(cfg: Config) -> dict[str, Any] | None:
    """Метаданные текущей прод-модели: версия, метрики, дата регистрации.

    TODO: MlflowClient().get_latest_versions(name, stages=['Production']).
    """
    raise NotImplementedError


def download_model(cfg: Config, stage: str = STAGE_PRODUCTION, dest: Path | None = None) -> Path:
    """Скачать артефакты модели локально (для инференса в контейнере).

    TODO: mlflow.artifacts.download_artifacts.
    """
    raise NotImplementedError
