"""Инференс: прогноз на 28 дней вперёд в витрину mart.forecast.

Фичи считаются ТЕМ ЖЕ кодом, что и на обучении — m5.features.build_features
с mode='inference'. Отдельного «скрипта для прода» нет и не должно быть.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from m5.config import Config
from m5.utils.logging import get_logger

logger = get_logger(__name__)


def predict(
    cfg: Config,
    as_of: date | str | None = None,
    model_stage: str = "Production",
    write_to_mart: bool = True,
) -> pd.DataFrame:
    """Прогноз на horizon дней вперёд от as_of.

    Args:
        as_of: последний день с фактическими продажами. По умолчанию — max(date) в stg.
        model_stage: какую версию брать из MLflow Model Registry
        write_to_mart: писать ли в mart.forecast

    Returns:
        (id, item_id, store_id, date, horizon, prediction, model_version, run_id, as_of)

    TODO:
        1. models = load_production_models(cfg, stage=model_stage)
        2. features = build_features(cfg, as_of=as_of, mode='inference')
        3. по каждой группе горизонтов взять свою модель и предсказать свои дни
        4. check_feature_consistency перед каждым predict
        5. post_process (клип по нулю, округление)
        6. sanity_checks -> при провале не писать в витрину и упасть
        7. write_forecast (идемпотентно, по ключу (id, date, as_of))
    """
    raise NotImplementedError


def post_process(preds: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Пост-обработка прогнозов.

    Что делаем:
        1. clip(0) — отрицательных продаж не бывает. Tweedie обычно не даёт
           отрицательных, но после любых поправок проверить надо.
        2. Занулить товары, которых нет в ассортименте (нет цены на неделю прогноза).
        3. Опционально: множитель-поправка на систематический bias по бэктесту.

    Округление до целых НЕ делаем: закупки агрегируют прогноз, и дробные
    значения при суммировании точнее.

    TODO: реализовать.
    """
    raise NotImplementedError


def sanity_checks(preds: pd.DataFrame, cfg: Config) -> dict[str, bool]:
    """Проверки перед записью в витрину. Прод-гейт, а не формальность.

    Проверяем:
        - нет NaN
        - нет отрицательных
        - число строк == n_series × horizon (ничего не потерялось)
        - суммарный объём прогноза в пределах ±40% от факта за прошлые 28 дней
        - нет дублей по (id, date)
        - все 28 горизонтов присутствуют

    Returns:
        {имя_проверки: прошла_ли}

    TODO: реализовать; при провале любой — не писать в витрину.
    """
    raise NotImplementedError


class SanityCheckError(RuntimeError):
    """Прогноз не прошёл проверки — в витрину не пишем."""


def write_forecast(preds: pd.DataFrame, cfg: Config) -> int:
    """Записать прогноз в mart.forecast идемпотентно.

    Идемпотентность обязательна: Airflow ретраит таск, и повторный запуск
    не должен задваивать строки. Реализуем через DELETE по (as_of) + INSERT
    в одной транзакции.

    Returns:
        Число записанных строк.

    TODO: BEGIN; DELETE FROM mart.forecast WHERE as_of = ?; INSERT ...; COMMIT;
    """
    raise NotImplementedError


def load_production_models(cfg: Config, stage: str = "Production") -> dict:
    """Забрать ансамбль моделей из MLflow Model Registry.

    TODO: mlflow.pyfunc / lightgbm.load_model по registered_model_name и stage.
    """
    raise NotImplementedError


def make_submission(preds: pd.DataFrame, cfg: Config, out_path: str | Path) -> Path:
    """Собрать sample_submission.csv для Kaggle (F1..F28 по каждому id).

    Нужно ровно для одного: чтобы сверить свой WRMSSE с публичным лидербордом
    и убедиться, что метрика посчитана правильно. В проде это не используется.

    TODO: pivot по горизонту в колонки F1..F28, дописать validation/evaluation-строки.
    """
    raise NotImplementedError
