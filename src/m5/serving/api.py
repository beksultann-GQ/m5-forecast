"""FastAPI поверх витрины прогнозов.

Осознанное архитектурное решение: API НЕ считает прогноз на лету.
Потребитель — отдел закупок с недельным циклом, realtime-инференс ему не нужен,
а батч-прогноз в витрине отдаётся за миллисекунды и не требует держать
модель и фичи в памяти сервиса.

Realtime-инференс здесь был бы карго-культом: сложнее, дороже, без пользы.
Это ровно тот разговор, который стоит вести на собеседовании.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from m5.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------- схемы ответов


class ForecastPoint(BaseModel):
    """Одна точка прогноза."""

    date: date
    horizon: int = Field(..., ge=1, le=28)
    prediction: float = Field(..., ge=0)


class ForecastResponse(BaseModel):
    """Прогноз по паре (item, store)."""

    item_id: str
    store_id: str
    as_of: date
    model_version: str
    forecast: list[ForecastPoint]


class HealthResponse(BaseModel):
    """Здоровье сервиса и свежесть данных."""

    status: str
    model_version: str | None
    forecast_as_of: date | None
    forecast_lag_days: int | None
    external_data_fresh: bool


# ---------------------------------------------------------------- приложение


def create_app() -> Any:
    """Собрать FastAPI-приложение.

    Эндпоинты:
        GET /health
            Живость + свежесть витрины. Если прогноз старше 2 дней — degraded.
            Это то, что дёргает мониторинг, а не человек.

        GET /forecast?item_id=&store_id=
            Прогноз на 28 дней по паре. 404, если пары нет в витрине.

        GET /forecast/store/{store_id}?date_from=&date_to=
            Прогноз по всему магазину — основной сценарий для закупок.

        GET /accuracy?level=&n_periods=
            История фактического качества из mart.forecast_accuracy.
            Потребитель должен видеть, насколько прогнозу можно верить.

        GET /metrics
            Prometheus-метрики: latency, RPS, свежесть витрины.

    TODO:
        1. FastAPI(title='M5 Forecast API', version=__version__)
        2. подключение к DuckDB в read_only на старте (lifespan)
        3. реализовать эндпоинты
        4. middleware с логированием времени ответа
    """
    raise NotImplementedError


# Для `uvicorn m5.serving.api:app`
app = None  # TODO: app = create_app()
