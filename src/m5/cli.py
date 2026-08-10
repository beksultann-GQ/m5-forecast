"""CLI — единая точка входа во всё. Airflow, Makefile и человек дёргают одно и то же.

Так не расходятся «как я запускаю локально» и «как это крутится в проде»:
DAG вызывает ровно те же команды, что и ты руками.

    m5 data download
    m5 data convert
    m5 data staging
    m5 data validate
    m5 external weather --mode archive
    m5 external holidays
    m5 features build --as-of 2016-04-24
    m5 model baseline
    m5 model train --as-of 2016-04-24
    m5 model backtest
    m5 model predict --as-of 2016-05-22
    m5 monitor drift
    m5 monitor quality
"""

from __future__ import annotations

from typing import Annotated

import typer

from m5.config import load_config
from m5.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

app = typer.Typer(
    name="m5",
    help="M5 Forecasting Accuracy — production-ready пайплайн",
    no_args_is_help=True,
    add_completion=False,
)

data_app = typer.Typer(name="data", help="Загрузка и подготовка данных", no_args_is_help=True)
external_app = typer.Typer(name="external", help="Внешние источники", no_args_is_help=True)
features_app = typer.Typer(name="features", help="Витрина фич", no_args_is_help=True)
model_app = typer.Typer(name="model", help="Обучение и инференс", no_args_is_help=True)
monitor_app = typer.Typer(name="monitor", help="Мониторинг", no_args_is_help=True)
db_app = typer.Typer(name="db", help="Заглянуть в DuckDB", no_args_is_help=True)

app.add_typer(data_app)
app.add_typer(external_app)
app.add_typer(features_app)
app.add_typer(model_app)
app.add_typer(monitor_app)
app.add_typer(db_app)

ConfigOpt = Annotated[str, typer.Option("--config", "-c", help="Каталог с yaml-конфигами")]
OverrideOpt = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Переопределить параметр: -s model.params.learning_rate=0.01"),
]
AsOfOpt = Annotated[
    str | None,
    typer.Option("--as-of", help="Граница знания, YYYY-MM-DD. По умолчанию max(date) в данных"),
]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Общие настройки."""
    from dotenv import load_dotenv

    from m5.utils.paths import repo_root

    # .env для Kaggle-кредов и MLflow. override=False: переменные окружения
    # сильнее файла — в Docker конфиг приходит именно через окружение.
    load_dotenv(repo_root() / ".env", override=False)
    setup_logging("DEBUG" if verbose else "INFO")


# ---------------------------------------------------------------- data


@data_app.command("download")
def data_download(
    config: ConfigOpt = "conf",
    force: Annotated[
        bool, typer.Option("--force", help="Перекачать, даже если файлы есть")
    ] = False,
) -> None:
    """Скачать датасет M5 с Kaggle."""
    from m5.data.download import download_competition

    cfg = load_config(config)
    path = download_competition(cfg, force=force)
    typer.echo(f"Данные в {path}")


@data_app.command("synth")
def data_synth(
    config: ConfigOpt = "conf",
    items: Annotated[int, typer.Option(help="Товаров на отдел")] = 20,
    # 1200 дней: хватает на 4 фолда rolling origin при min_train_days=730
    days: Annotated[int, typer.Option(help="Длина истории в днях")] = 1200,
    seed: int = 42,
) -> None:
    """Сгенерировать синтетический M5 той же формы — чтобы прогнать пайплайн без Kaggle."""
    from m5.data.synth import generate

    cfg = load_config(config)
    paths = generate(cfg, n_items_per_dept=items, n_days=days, seed=seed)
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


@data_app.command("convert")
def data_convert(config: ConfigOpt = "conf", overwrite: bool = False) -> None:
    """CSV -> Parquet (raw-слой)."""
    from m5.data.convert import csv_to_parquet

    cfg = load_config(config)
    result = csv_to_parquet(cfg, overwrite=overwrite)
    for name, path in result.items():
        typer.echo(f"{name}: {path}")


@data_app.command("staging")
def data_staging(config: ConfigOpt = "conf") -> None:
    """wide -> long + справочники (DuckDB, схема stg)."""
    from m5.data.staging import build_staging

    cfg = load_config(config)
    stats = build_staging(cfg)
    for table, n_rows in stats.items():
        typer.echo(f"{table}: {n_rows:,} строк")


@data_app.command("validate")
def data_validate(
    config: ConfigOpt = "conf",
    table: Annotated[str, typer.Option(help="Какую таблицу проверять")] = "stg.sales_enriched",
    schema: Annotated[str, typer.Option(help="Имя схемы из m5.data.schemas")] = "sales_long",
) -> None:
    """Проверить данные (pandera + SQL-инварианты). Падает при нарушении."""
    from m5.data.duck import connect
    from m5.data.schemas import validate_duckdb_table

    cfg = load_config(config)
    with connect(cfg.paths.duckdb_path, read_only=True) as con:
        report = validate_duckdb_table(con, table, schema)
    typer.echo(report)


# ---------------------------------------------------------------- external


@external_app.command("weather")
def external_weather(
    config: ConfigOpt = "conf",
    mode: Annotated[str, typer.Option(help="archive | forecast | full")] = "archive",
    offline: Annotated[bool, typer.Option(help="Только кеш, в сеть не ходить")] = False,
) -> None:
    """Погода из Open-Meteo (или Visual Crossing) -> data/external/weather.parquet."""
    from m5.external.weather import fetch_weather

    cfg = load_config(config)
    path = fetch_weather(cfg, mode=mode, offline=offline)
    typer.echo(f"Погода: {path}")


@external_app.command("holidays")
def external_holidays(config: ConfigOpt = "conf", offline: bool = False) -> None:
    """Госпраздники США из Nager.Date -> data/external/holidays.parquet."""
    from m5.external.holidays import fetch_holidays

    cfg = load_config(config)
    path = fetch_holidays(cfg, offline=offline)
    typer.echo(f"Праздники: {path}")


@external_app.command("compare-events")
def external_compare(config: ConfigOpt = "conf") -> None:
    """Сверить праздники Nager.Date с event_name из M5. Диагностика для README."""
    from m5.external.holidays import compare_with_m5_events

    load_config(config)
    raise NotImplementedError(compare_with_m5_events.__doc__)


# ---------------------------------------------------------------- features


@features_app.command("build")
def features_build(
    config: ConfigOpt = "conf",
    as_of: AsOfOpt = None,
    mode: Annotated[str, typer.Option(help="train | inference")] = "train",
    stores: Annotated[
        list[str] | None,
        typer.Option("--store", help="Собрать только по этим магазинам: --store CA_1 --store CA_2"),
    ] = None,
) -> None:
    """Собрать витрину фич по состоянию на as_of."""
    from m5.features.build import build_features

    cfg = load_config(config)
    stats = build_features(cfg, as_of=as_of, mode=mode, stores=stores)
    for table, n_rows in stats.items():
        typer.echo(f"{table}: {n_rows:,} строк")


# ---------------------------------------------------------------- model


@model_app.command("baseline")
def model_baseline(config: ConfigOpt = "conf") -> None:
    """Посчитать baseline'ы. Планка, которую обязана побить модель."""
    from m5.models.baseline import run_baselines

    cfg = load_config(config)
    typer.echo(run_baselines(cfg).to_string())


@model_app.command("train")
def model_train(
    config: ConfigOpt = "conf",
    as_of: AsOfOpt = None,
    overrides: OverrideOpt = None,
    stores: Annotated[
        list[str] | None, typer.Option("--store", help="Обучить только по этим магазинам")
    ] = None,
    no_mlflow: Annotated[bool, typer.Option("--no-mlflow")] = False,
) -> None:
    """Обучить ансамбль LightGBM (по модели на магазин)."""
    from m5.models.train import feature_importance, train

    cfg = load_config(config, overrides=overrides)
    models = train(cfg, as_of=as_of, log_to_mlflow=not no_mlflow, stores=stores)
    typer.echo(f"Обучено моделей: {len(models)}\n")
    typer.echo("Топ-15 фич по вкладу (gain):")
    typer.echo(feature_importance(models, top_n=15).to_string(index=False))


@model_app.command("importance")
def model_importance(
    config: ConfigOpt = "conf",
    top: Annotated[int, typer.Option(help="Сколько фич показать")] = 30,
) -> None:
    """Показать важность фич последнего обучения — на что модель реально смотрит."""
    from pathlib import Path

    import pandas as pd

    cfg = load_config(config)
    path = Path(cfg.paths.processed_dir) / "models" / "latest" / "feature_importance.csv"
    if not path.exists():
        raise typer.BadParameter(f"Нет {path}. Сначала `make train`")

    frame = pd.read_csv(path).head(top)
    frame["gain_share"] = (frame["gain_share"] * 100).round(2).astype(str) + "%"
    typer.echo(frame[["feature", "gain", "split", "gain_share"]].to_string(index=False))


@model_app.command("backtest")
def model_backtest(
    config: ConfigOpt = "conf",
    overrides: OverrideOpt = None,
    ablation: Annotated[bool, typer.Option(help="Ablation по внешним данным")] = False,
) -> None:
    """Rolling-origin бэктест: WRMSSE по фолдам + сравнение с baseline."""
    from m5.evaluation.backtest import ablation_external_data, run_backtest

    cfg = load_config(config, overrides=overrides)
    if ablation:
        typer.echo(ablation_external_data(cfg).to_string())
        return
    result = run_backtest(cfg)
    typer.echo(result.to_frame().to_string())
    typer.echo(f"Побили baseline: {result.beats_baseline()}")


@model_app.command("predict")
def model_predict(
    config: ConfigOpt = "conf",
    as_of: AsOfOpt = None,
    stage: Annotated[str, typer.Option(help="Стадия модели в registry")] = "Production",
    submission: Annotated[str | None, typer.Option(help="Путь для Kaggle submission.csv")] = None,
    no_write: Annotated[
        bool, typer.Option("--no-write", help="Посчитать и проверить, но не публиковать")
    ] = False,
) -> None:
    """Прогноз на 28 дней -> mart.forecast.

    Sanity-checks выполняются ВНУТРИ и гейтят запись: при провале любой проверки
    команда падает, а витрина остаётся нетронутой (fail closed).
    """
    from m5.models.predict import make_submission, predict

    cfg = load_config(config)
    preds = predict(cfg, as_of=as_of, model_stage=stage, write_to_mart=not no_write)
    typer.echo(f"Прогнозов: {len(preds):,}")
    if submission:
        typer.echo(f"Submission: {make_submission(preds, cfg, submission)}")


#: Код выхода, по которому Airflow помечает таск как skipped, а не failed.
#: Это дефолт BashOperator.skip_on_exit_code.
SKIP_EXIT_CODE = 99


@model_app.command("gate")
def model_gate(
    config: ConfigOpt = "conf",
    version: Annotated[int | None, typer.Option(help="Версия-кандидат")] = None,
    metric: Annotated[str, typer.Option(help="По какой метрике сравнивать")] = "wmape",
    min_improvement: Annotated[float, typer.Option()] = 0.01,
) -> None:
    """Гейт промоушена: решить, лучше ли кандидат прод-модели. Ничего не меняет.

    Код выхода 0 — кандидат лучше, идём регистрировать.
    Код выхода 99 — не лучше; Airflow помечает таск skipped, и ветка ниже
    не выполняется. «Модель не стала лучше» — нормальный исход недели,
    а не авария, поэтому не failed.
    """
    from m5.models.registry import compare_with_production

    cfg = load_config(config)
    better, reason, _ = compare_with_production(cfg, version, metric, min_improvement)
    typer.echo(reason)

    if not better:
        raise typer.Exit(code=SKIP_EXIT_CODE)


@model_app.command("promote")
def model_promote(
    config: ConfigOpt = "conf",
    version: Annotated[
        int | None, typer.Option(help="Версия-кандидат. По умолчанию — последняя")
    ] = None,
    metric: Annotated[str, typer.Option(help="По какой метрике сравнивать")] = "wmape",
    min_improvement: Annotated[
        float, typer.Option(help="Минимальное относительное улучшение")
    ] = 0.01,
) -> None:
    """Промоутить версию в Production, только если она лучше текущей прод-модели."""
    from m5.models.registry import promote_if_better

    cfg = load_config(config)
    promoted = promote_if_better(cfg, version, metric=metric, min_improvement=min_improvement)
    typer.echo("Промоутнули в Production" if promoted else "Прод-модель осталась прежней")


# ---------------------------------------------------------------- monitor


@monitor_app.command("drift")
def monitor_drift(config: ConfigOpt = "conf") -> None:
    """Проверить дрифт данных и прогнозов."""
    from m5.monitoring.drift import check_drift_and_alert

    cfg = load_config(config)
    typer.echo(f"Нужно переобучение: {check_drift_and_alert(cfg)}")


@monitor_app.command("quality")
def monitor_quality(config: ConfigOpt = "conf", as_of: AsOfOpt = None) -> None:
    """Посчитать фактическое качество по приехавшим продажам."""
    from m5.monitoring.quality import compute_actual_accuracy

    cfg = load_config(config)
    typer.echo(compute_actual_accuracy(cfg, as_of=as_of).to_string())


# ---------------------------------------------------------------- db


@db_app.command("path")
def db_path(config: ConfigOpt = "conf") -> None:
    """Путь к файлу DuckDB — его и надо указать в DBeaver."""
    cfg = load_config(config)
    typer.echo(cfg.paths.duckdb_path)


@db_app.command("tables")
def db_tables(config: ConfigOpt = "conf") -> None:
    """Список таблиц по слоям с числом строк."""
    from m5.data.duck import connect

    cfg = load_config(config)
    with connect(cfg.paths.duckdb_path, read_only=True) as con:
        rows = con.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
            ORDER BY table_schema, table_name
            """
        ).fetchall()
        for schema, table in rows:
            n_rows = con.execute(f'SELECT COUNT(*) FROM "{schema}"."{table}"').fetchone()[0]
            n_cols = len(con.execute(f'DESCRIBE "{schema}"."{table}"').fetchall())
            typer.echo(f"{schema}.{table:<20} {n_rows:>12,} строк  {n_cols:>3} колонок")


@db_app.command("peek")
def db_peek(
    table: Annotated[str, typer.Argument(help="Например ft.mart")],
    config: ConfigOpt = "conf",
    limit: int = 10,
    columns: Annotated[str, typer.Option(help="Список колонок через запятую")] = "*",
) -> None:
    """Показать первые строки таблицы прямо в терминале."""
    from m5.data.duck import connect

    cfg = load_config(config)
    with connect(cfg.paths.duckdb_path, read_only=True) as con:
        frame = con.execute(f"SELECT {columns} FROM {table} LIMIT {limit}").df()
    typer.echo(frame.to_string())


@db_app.command("sql")
def db_sql(
    query: Annotated[str, typer.Argument(help="Произвольный SELECT")],
    config: ConfigOpt = "conf",
) -> None:
    """Выполнить SELECT (read-only) — быстрее, чем открывать DBeaver."""
    from m5.data.duck import connect

    cfg = load_config(config)
    with connect(cfg.paths.duckdb_path, read_only=True) as con:
        typer.echo(con.execute(query).df().to_string())


if __name__ == "__main__":
    app()
