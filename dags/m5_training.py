"""DAG обучения. Раз в неделю, по воскресеньям.

    ingest -> validate -> external -> build_features -> train -> backtest
          -> сравнить с прод-моделью -> зарегистрировать, ЕСЛИ лучше

Ключевой момент — последний шаг. Новая модель НЕ едет в прод автоматически
просто потому, что она новая. Гейт: WRMSSE на бэктесте должен быть лучше
текущей прод-модели минимум на 1%. Меньше — это шум между фолдами.
"""

from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.utils.task_group import TaskGroup
from common import DEFAULT_ARGS, alert_on_failure, alert_on_sla_miss, m5_command

DAG_ID = "m5_training"

with DAG(
    dag_id=DAG_ID,
    description="Еженедельное переобучение модели прогноза продаж M5",
    default_args=DEFAULT_ARGS,
    schedule="0 2 * * 0",  # воскресенье, 02:00
    start_date=datetime(2024, 1, 1),
    catchup=False,  # переобучаться «задним числом» смысла нет
    max_active_runs=1,  # два обучения одновременно память не переживёт
    sla_miss_callback=alert_on_sla_miss,
    tags=["m5", "ml", "training"],
) as dag:
    start = EmptyOperator(task_id="start")

    # ------------------------------------------------------------ данные
    with TaskGroup("ingest") as ingest:
        download = m5_command("download", "data download")
        convert = m5_command("convert", "data convert")
        staging = m5_command("staging", "data staging")
        download >> convert >> staging

    # Валидация — жёсткий гейт. Битые данные дальше не идут.
    validate = m5_command(
        "validate",
        "data validate --table stg.sales_enriched --schema sales_long",
        retries=0,  # ретраить бессмысленно: данные не «починятся» сами
    )

    # ------------------------------------------------------------ внешние источники
    # Погода и праздники тянутся параллельно: они независимы.
    # Падение внешнего API не должно ронять обучение целиком —
    # trigger_rule ниже позволяет продолжить на закешированных данных.
    with TaskGroup("external") as external:
        weather = m5_command("weather", "external weather --mode archive")
        holidays = m5_command("holidays", "external holidays")

    # ------------------------------------------------------------ фичи и обучение
    build_features = m5_command(
        "build_features",
        "features build --as-of {{ ds }} --mode train",
        trigger_rule="all_done",  # внешние данные опциональны, ядро пайплайна — нет
    )

    train = m5_command(
        "train",
        "model train --as-of {{ ds }}",
        execution_timeout=None,
        pool="training_pool",  # ограничиваем параллелизм: обучение жрёт память
    )

    backtest = m5_command("backtest", "model backtest")

    # ------------------------------------------------------------ гейт и регистрация
    def _beats_production(**context) -> bool:
        """Сравнить кандидата с прод-моделью.

        Returns:
            True -> идём регистрировать, False -> DAG корректно останавливается.

        TODO:
            1. взять метрики последнего backtest-run из MLflow
            2. взять метрики текущей Production-версии
            3. вернуть candidate_wrmsse < prod_wrmsse * (1 - min_improvement)
            4. записать решение и причину в XCom — чтобы было видно в UI
        """
        raise NotImplementedError

    is_better = ShortCircuitOperator(
        task_id="is_better_than_production",
        python_callable=_beats_production,
        on_failure_callback=alert_on_failure,
    )

    register = m5_command(
        "register_model", "model promote --version {{ ti.xcom_pull(key='version') }}"
    )

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    start >> ingest >> validate >> external >> build_features
    build_features >> train >> backtest >> is_better >> register >> end
