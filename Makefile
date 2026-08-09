.DEFAULT_GOAL := help
SHELL := /bin/bash
UV := uv run

.PHONY: help
help:  ## Показать доступные команды
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- окружение
.PHONY: setup
setup:  ## Создать venv и поставить зависимости (uv)
	uv python install 3.12
	uv sync --all-extras
	$(UV) pre-commit install

.PHONY: lint
lint:  ## ruff check
	$(UV) ruff check src tests dags

.PHONY: fmt
fmt:  ## ruff format + автофиксы
	$(UV) ruff format src tests dags
	$(UV) ruff check --fix src tests dags

.PHONY: test
test:  ## pytest (без сетевых и медленных)
	$(UV) pytest -m "not network and not slow"

.PHONY: test-all
test-all:  ## pytest целиком
	$(UV) pytest

# ---------------------------------------------------------------- данные
.PHONY: synth
synth:  ## Сгенерировать синтетический M5 (без Kaggle) — для демо и CI
	$(UV) m5 data synth

.PHONY: download
download:  ## Скачать датасет M5 с Kaggle в data/raw
	$(UV) m5 data download

.PHONY: raw
raw:  ## CSV -> Parquet (immutable raw-слой)
	$(UV) m5 data convert

.PHONY: staging
staging:  ## wide -> long в DuckDB (staging-слой)
	$(UV) m5 data staging

.PHONY: validate
validate:  ## Валидация схем (pandera)
	$(UV) m5 data validate

.PHONY: external
external:  ## Подтянуть погоду (Open-Meteo) и праздники (Nager.Date)
	$(UV) m5 external weather
	$(UV) m5 external holidays

.PHONY: features
features:  ## Собрать витрину фич (DuckDB SQL)
	$(UV) m5 features build

# ---------------------------------------------------------------- модель
.PHONY: baseline
baseline:  ## Посчитать seasonal naive baseline
	$(UV) m5 model baseline

.PHONY: train
train:  ## Обучить LightGBM (tweedie)
	$(UV) m5 model train

.PHONY: backtest
backtest:  ## Rolling-origin бэктест + WRMSSE
	$(UV) m5 model backtest

.PHONY: predict
predict:  ## Прогноз на 28 дней в витрину
	$(UV) m5 model predict

# ---------------------------------------------------------------- сервисы
.PHONY: up-mlflow
up-mlflow:  ## Поднять только postgres + mlflow (лёгкий шаг, без Airflow)
	docker compose -f docker/docker-compose.yml up -d postgres mlflow

.PHONY: up
up:  ## Поднять airflow + mlflow + postgres
	docker compose -f docker/docker-compose.yml up -d

.PHONY: down
down:  ## Остановить всё
	docker compose -f docker/docker-compose.yml down

.PHONY: logs
logs:  ## Логи docker compose
	docker compose -f docker/docker-compose.yml logs -f

.PHONY: mlflow
mlflow:  ## Локальный MLflow UI (без докера)
	$(UV) mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

.PHONY: api
api:  ## FastAPI-сервис прогнозов
	$(UV) uvicorn m5.serving.api:app --reload --port 8000

.PHONY: dashboard
dashboard:  ## Streamlit-дашборд мониторинга
	$(UV) streamlit run dashboards/streamlit_app.py

# ---------------------------------------------------------------- пайплайн
.PHONY: tables
tables:  ## Список таблиц в DuckDB с числом строк
	$(UV) m5 db tables

.PHONY: dbpath
dbpath:  ## Путь к файлу DuckDB (его указать в DBeaver)
	@$(UV) m5 db path

.PHONY: demo
demo: synth raw staging features tables  ## Синтетика -> витрина за один заход, без Kaggle

.PHONY: pipeline
pipeline: raw staging validate external features train backtest  ## Тонкая вертикаль целиком

.PHONY: clean
clean:  ## Удалить производные артефакты (raw не трогаем)
	rm -rf data/interim data/processed .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
