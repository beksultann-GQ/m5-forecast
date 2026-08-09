"""HTTP-клиент для внешних API: ретраи, таймауты, кеш на диске.

Зачем кеш: история погоды за 2011-2016 не меняется. Дёргать Open-Meteo на каждом
прогоне DAG'а — бессмысленная нагрузка на бесплатный API и риск rate-limit.
Кешируем ответ по хешу (url + params), инвалидируем только для будущих дат.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from m5.utils.logging import get_logger
from m5.utils.paths import ensure_dir

logger = get_logger(__name__)

RETRYABLE = (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)


class ExternalAPIError(RuntimeError):
    """Внешний API недоступен или вернул мусор."""


class CachedJSONClient:
    """GET-клиент с файловым кешем ответов.

    Args:
        cache_dir: куда складывать .json-ответы
        timeout: таймаут запроса, сек
        max_retries: сколько раз ретраить 5xx/429/таймауты
        offline: если True — ходить только в кеш, при промахе падать.
            Нужно в CI и в тестах: пайплайн не должен зависеть от сети.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_sec: float = 2.0,
        offline: bool = False,
        user_agent: str = "m5-forecast/0.1 (+educational project)",
    ) -> None:
        self.cache_dir = ensure_dir(cache_dir)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.offline = offline
        self.headers = {"User-Agent": user_agent, "Accept": "application/json"}

    # ------------------------------------------------------------------ кеш
    def cache_key(self, url: str, params: dict[str, Any]) -> str:
        """Стабильный ключ: sha256 от url + отсортированных params."""
        payload = json.dumps({"url": url, "params": params}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def cache_path(self, url: str, params: dict[str, Any]) -> Path:
        return self.cache_dir / f"{self.cache_key(url, params)}.json"

    def read_cache(self, url: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Достать ответ из кеша.

        TODO: прочитать json, если файл есть; вернуть None при промахе.
        """
        raise NotImplementedError

    def write_cache(self, url: str, params: dict[str, Any], payload: dict[str, Any]) -> Path:
        """Сохранить ответ в кеш.

        TODO: записать json атомарно (во временный файл + rename).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ сеть
    @retry(
        retry=retry_if_exception_type(RETRYABLE),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        """Один GET с ретраями. Не вызывать напрямую — только через get()."""
        with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    def get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """GET с кешем и ретраями.

        TODO:
            1. params = params or {}
            2. если use_cache -> read_cache(); попали — вернуть
            3. если self.offline -> ExternalAPIError('промах кеша в offline-режиме')
            4. payload = self._get(...)
            5. write_cache(), вернуть payload
        """
        raise NotImplementedError

    def get_many(
        self,
        requests: list[tuple[str, dict[str, Any]]],
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Последовательные GET'ы (Open-Meteo не любит параллель с бесплатного тарифа).

        TODO: пройти списком, собрать результаты, логировать прогресс каждые 10 запросов.
        """
        raise NotImplementedError
