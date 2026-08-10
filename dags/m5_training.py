"""DAG обучения. Раз в неделю, по воскресеньям.

    ingest -> validate -> external -> build_features -> train -> backtest
          -> сравнить с прод-моделью -> зарегистрировать, ЕСЛИ лучше

Ключевой момент — последний шаг. Новая модель НЕ едет в прод автоматически
просто потому, что она новая. Гейт: WRMSSE на бэктесте должен быть лучше
текущей прод-модели минимум на 1%. Меньше — это шум между фолдами.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from common import DEFAULT_ARGS, alert_on_sla_miss, m5_command

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
    # retries=0 здесь осознанно, хотя обычно у сетевых задач ретраи нужны.
    # Причина: внутри ЭТОГО DAG'а внешние данные опциональны — ниже стоит
    # trigger_rule="all_done", и обучение пойдёт даже без них. При retries=3
    # с экспоненциальной паузой (5 → 10 → 20 мин) необязательный шаг держал бы
    # весь пайплайн заложником на полчаса.
    # Настоящие ретраи живут в m5_external_ingest, где загрузка — самоцель.
    with TaskGroup("external") as external:
        weather = m5_command(
            "weather",
            "external weather --mode archive",
            retries=0,
            execution_timeout=timedelta(minutes=20),
        )
        holidays = m5_command(
            "holidays",
            "external holidays",
            retries=0,
            execution_timeout=timedelta(minutes=10),
        )

    # ------------------------------------------------------------ фичи и обучение
    build_features = m5_command(
        "build_features",
        "features build --as-of {{ ds }} --mode train",
        trigger_rule="all_done",  # внешние данные опциональны, ядро пайплайна — нет
    )

    # {{ ds }} — календарная дата запуска. В реальном проде это и есть граница
    # знания. На историческом датасете (M5 кончается 2016-06-19) ds уезжает
    # в будущее, поэтому m5.models.train.clamp_as_of обрезает его до последней
    # даты с фактом и громко предупреждает. Без обрезки валидационное окно
    # оказывается пустым, а обучение — молча без early stopping и без метрик.
    train = m5_command(
        "train",
        "model train --as-of {{ ds }}",
        execution_timeout=None,
        pool="training_pool",  # ограничиваем параллелизм: обучение жрёт память
    )

    backtest = m5_command("backtest", "model backtest")

    # ------------------------------------------------------------ гейт и регистрация
    #
    # Гейт — обычный BashOperator, а не ShortCircuitOperator с python_callable.
    # Это не стилистика, а необходимость: PythonOperator исполняется в процессе,
    # который Airflow форкает от многопоточного воркера. Импорт mlflow и LightGBM
    # в таком форке на Linux валит процесс молча, без питоновского traceback —
    # в логе остаётся только «exited with return code 1», и искать причину
    # приходится часами.
    #
    # Заодно это возвращает нас к принципу проекта: DAG вызывает те же команды
    # `m5`, что и человек в терминале. Никакой отдельной логики в DAG'е нет.
    #
    # skip_on_exit_code=99: `m5 model gate` выходит с этим кодом, когда кандидат
    # не лучше прод-модели. Таск помечается skipped, ветка ниже не выполняется,
    # DAG завершается зелёным. «Модель не стала лучше» — нормальный исход недели.
    is_better = m5_command(
        "is_better_than_production",
        "model gate --metric wmape",
        skip_on_exit_code=99,
        retries=0,  # сравнение детерминировано, ретраить нечего
    )

    # promote перепроверяет условие сам — это делает таск идемпотентным:
    # повторный запуск при ретрае не промоутит модель второй раз вслепую.
    register = m5_command("register_model", "model promote")

    # none_failed, а не none_failed_min_one_success: когда гейт решил не
    # промоутить, register_model становится skipped, и требование «хотя бы один
    # успешный апстрим» не выполнялось бы. Но «модель не стала лучше» — штатный
    # исход, и терминальный узел обязан отработать.
    end = EmptyOperator(task_id="end", trigger_rule="none_failed")

    start >> ingest >> validate >> external >> build_features
    build_features >> train >> backtest >> is_better >> register >> end
