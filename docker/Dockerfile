# Базовый образ проекта: пакет m5 + CLI. Используется API, дашбордом и Airflow.
# На Apple Silicon собирается под arm64 нативно — не ставьте --platform linux/amd64,
# LightGBM под эмуляцией работает в разы медленнее.

FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    M5_REPO_ROOT=/opt/m5

# libgomp — рантайм OpenMP, без него LightGBM не импортируется
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /opt/m5

# ---------------------------------------------------------------- зависимости
# Отдельным слоем: меняются редко, кешируются между сборками кода
FROM base AS deps
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-install-project --all-extras || uv sync --no-install-project --all-extras

# ---------------------------------------------------------------- приложение
FROM deps AS app
COPY src/ ./src/
COPY conf/ ./conf/
COPY dags/ ./dags/
COPY README.md Makefile ./
RUN uv sync --all-extras

ENV PATH="/opt/m5/.venv/bin:$PATH"

# Непривилегированный пользователь
RUN useradd --create-home --uid 1000 m5 && chown -R m5:m5 /opt/m5
USER m5

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import m5; print(m5.__version__)" || exit 1

ENTRYPOINT ["m5"]
CMD ["--help"]

# ---------------------------------------------------------------- API
FROM app AS api
EXPOSE 8000
ENTRYPOINT ["uvicorn"]
CMD ["m5.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------- дашборд
FROM app AS dashboard
EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "dashboards/streamlit_app.py", \
            "--server.address=0.0.0.0", "--server.port=8501"]
