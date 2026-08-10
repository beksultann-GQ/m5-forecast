"""Общее для всех DAG'ов: дефолты, алерты, обёртка над CLI.

Принцип: DAG не содержит бизнес-логики. Он вызывает те же команды `m5 ...`,
что и человек локально. Логика живёт в пакете, DAG — только расписание,
ретраи и зависимости. Так «работает у меня» и «работает в проде» не расходятся.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from airflow.operators.bash import BashOperator

# Корень проекта внутри контейнера Airflow
PROJECT_DIR = "/opt/m5"

DEFAULT_ARGS: dict[str, Any] = {
    "owner": "ml-platform",
    "depends_on_past": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "email_on_failure": False,  # алерты шлём через on_failure_callback
    "execution_timeout": timedelta(hours=3),
}


def alert_on_failure(context: dict[str, Any]) -> None:
    """Колбэк на падение таска.

    Сообщение должно быть самодостаточным: дежурный не должен лезть в код,
    чтобы понять, что сломалось.

    TODO: отправить в Slack/telegram: dag_id, task_id, execution_date,
          ссылка на лог, текст исключения.
    """
    task = context["task_instance"]
    message = (
        f"[M5] Упал таск {task.dag_id}.{task.task_id}\n"
        f"execution_date: {context['ds']}\n"
        f"лог: {task.log_url}"
    )
    print(message)


def alert_on_sla_miss(*args: Any, **kwargs: Any) -> None:
    """Колбэк на нарушение SLA.

    TODO: отправить предупреждение (не критично, но требует внимания).
    """
    print("[M5] SLA miss")


def m5_command(
    task_id: str,
    command: str,
    dag: Any = None,
    pool: str | None = None,
    **kwargs: Any,
) -> BashOperator:
    """Таск, вызывающий CLI `m5`.

    Args:
        command: то, что идёт после `m5`, например 'features build --as-of {{ ds }}'

    Все таски должны быть ИДЕМПОТЕНТНЫ: повторный запуск с тем же {{ ds }}
    обязан давать тот же результат, а не задваивать данные. Это обеспечивается
    на уровне SQL (CREATE OR REPLACE, DELETE+INSERT по ключу), а не здесь.

    Вызываем `m5`, а не `uv run m5`: в образе Airflow пакет ставится прямо
    в системное окружение (см. docker/Dockerfile.airflow), никакого uv там нет.
    """
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd {PROJECT_DIR} && m5 {command}",
        on_failure_callback=alert_on_failure,
        dag=dag,
        pool=pool,
        **kwargs,
    )
