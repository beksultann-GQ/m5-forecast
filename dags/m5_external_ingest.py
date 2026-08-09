"""DAG загрузки внешних данных. Ежедневно, отдельно от обучения и инференса.

Почему отдельным DAG'ом, а не таском внутри обучения:
    1. Внешние API падают по своим причинам. Их падение не должно ронять
       ни обучение, ни прогноз — те могут отработать на вчерашнем кеше.
    2. Разные расписания: погоду тянем каждый день, праздники — раз в год.
    3. Ретраи с длинными паузами: бесплатные API лучше не долбить.

Праздники (Nager.Date) обновляются раз в год — календарь 2011 года
уже не изменится. Гоняем раз в неделю просто чтобы поймать появление
данных на следующий год.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from common import DEFAULT_ARGS, alert_on_failure, m5_command

DAG_ID = "m5_external_ingest"

EXTERNAL_ARGS = {
    **DEFAULT_ARGS,
    "retries": 5,
    "retry_delay": timedelta(minutes=15),  # внешние API — ретраим редко и терпеливо
}

with DAG(
    dag_id=DAG_ID,
    description="Погода (Open-Meteo) и госпраздники (Nager.Date)",
    default_args=EXTERNAL_ARGS,
    schedule="0 3 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["m5", "external", "ingest"],
) as dag:
    start = EmptyOperator(task_id="start")

    with TaskGroup("weather") as weather_group:
        # История: догружаем хвост, который добавился в archive (лаг ~5 дней).
        # Кеш делает это дешёвым — уже скачанные годы повторно не тянутся.
        archive = m5_command("weather_archive", "external weather --mode archive")

        # Прогноз: 16 дней вперёд, кешировать нельзя — он обновляется.
        forecast = m5_command("weather_forecast", "external weather --mode forecast")

        archive >> forecast

    # Праздники: тяжёлых ретраев не нужно, эндпоинт лёгкий и стабильный.
    holidays = m5_command("holidays", "external holidays")

    def _validate_external(**context) -> None:
        """Проверить свежесть и полноту внешних данных.

        Проверяем:
            - weather: max(date) не старше 7 дней, нет дыр в датах по каждому штату,
              температура в физически осмысленных пределах
            - holidays: покрыт весь диапазон лет, есть все три субдивизии

        Мягкая деградация: если погода протухла, помечаем и продолжаем
        (модель отработает на климатической норме), но алерт шлём.

        TODO: вызвать m5.monitoring.drift.external_data_freshness и
              m5.data.schemas.validate по weather/holidays.
        """
        raise NotImplementedError

    validate = PythonOperator(
        task_id="validate_external",
        python_callable=_validate_external,
        on_failure_callback=alert_on_failure,
    )

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    start >> [weather_group, holidays] >> validate >> end
