"""DAG инференса. Ежедневно.

    забрать модель из registry -> посчитать фичи -> предсказать на 28 дней
        -> sanity-checks -> записать в витрину -> посчитать факт. качество

Порядок важен: sanity-checks ДО записи в витрину. Плохой прогноз лучше
не показать вообще, чем показать: отдел закупок по нему закажет товар.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.sensors.external_task import ExternalTaskSensor
from common import DEFAULT_ARGS, alert_on_sla_miss, m5_command

DAG_ID = "m5_inference"

with DAG(
    dag_id=DAG_ID,
    description="Ежедневный батч-прогноз продаж на 28 дней",
    default_args=DEFAULT_ARGS,
    schedule="0 5 * * *",  # ежедневно, 05:00 — до начала рабочего дня
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    sla_miss_callback=alert_on_sla_miss,
    tags=["m5", "ml", "inference"],
) as dag:
    start = EmptyOperator(task_id="start")

    # Свежие данные о продажах должны доехать раньше прогноза.
    # В реальном проде здесь был бы сенсор на витрину источника.
    wait_for_data = ExternalTaskSensor(
        task_id="wait_for_sales_data",
        external_dag_id="m5_external_ingest",
        external_task_id="end",
        execution_delta=timedelta(hours=2),
        mode="reschedule",  # не занимаем слот воркера, пока ждём
        timeout=60 * 60 * 2,
        poke_interval=300,
        soft_fail=False,
    )

    # Погода на будущее: archive отстаёт на ~5 дней, поэтому берём forecast.
    # Дни 17-28 горизонта закроются климатической нормой — прогноза туда нет.
    weather_forecast = m5_command(
        "weather_forecast",
        "external weather --mode full",
        retries=2,
        # Если Open-Meteo лёг — не роняем прогноз целиком: климатическая норма
        # хуже прогноза, но лучше отсутствия прогноза.
        trigger_rule="all_success",
    )

    build_features = m5_command(
        "build_features",
        "features build --as-of {{ ds }} --mode inference",
        trigger_rule="all_done",
    )

    # Прогноз и публикация — ОДИН таск, а не два.
    #
    # Соблазн разнести «посчитать» и «опубликовать» на разные узлы графа велик:
    # так нагляднее. Но тогда прогноз считался бы дважды (полный инференс —
    # не бесплатная операция), а между проверкой и записью появлялось бы окно,
    # в котором данные могли измениться.
    #
    # Sanity-checks живут внутри `m5 model predict`: они выполняются ДО записи
    # и при провале валят команду, оставляя витрину нетронутой. Fail closed
    # обеспечивается в коде, а не порядком узлов в графе.
    forecast = m5_command(
        "predict_and_publish",
        "model predict --as-of {{ ds }} --stage Production",
        retries=1,  # один ретрай на случай сетевого сбоя при загрузке модели
    )

    # Фактическое качество прогноза, сделанного 28 дней назад:
    # только сегодня по нему приехал полный факт.
    actual_quality = m5_command("actual_quality", "monitor quality --as-of {{ ds }}")

    drift = m5_command("drift_check", "monitor drift")

    end = EmptyOperator(task_id="end")

    start >> wait_for_data >> weather_forecast >> build_features
    build_features >> forecast >> [actual_quality, drift] >> end
