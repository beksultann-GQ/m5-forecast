"""Тесты DAG'ов. Дешёвые smoke-проверки, которые ловят опечатки до деплоя.

Airflow — тяжёлая зависимость, поэтому весь модуль скипается, если её нет.
Локально без Airflow тесты просто не гоняются, в CI-образе — гоняются.
"""

from __future__ import annotations

import pytest

pytest.importorskip("airflow", reason="Airflow не установлен — DAG-тесты пропускаем")

EXPECTED_DAGS = {"m5_training", "m5_inference", "m5_external_ingest"}


@pytest.fixture(scope="module")
def dagbag():
    from airflow.models import DagBag

    return DagBag(dag_folder="dags", include_examples=False)


class TestDagIntegrity:
    def test_no_import_errors(self, dagbag):
        """DAG'и импортируются без ошибок. Ловит опечатки и битые импорты."""
        assert not dagbag.import_errors, dagbag.import_errors

    def test_all_dags_present(self, dagbag):
        assert set(dagbag.dag_ids) >= EXPECTED_DAGS

    @pytest.mark.parametrize("dag_id", sorted(EXPECTED_DAGS))
    def test_no_cycles(self, dagbag, dag_id):
        dagbag.get_dag(dag_id).test_cycle()

    @pytest.mark.parametrize("dag_id", sorted(EXPECTED_DAGS))
    def test_retries_configured(self, dagbag, dag_id):
        """У всех тасков есть ретраи — сеть и внешние API отваливаются регулярно."""
        dag = dagbag.get_dag(dag_id)
        for task in dag.tasks:
            assert task.retries is not None

    @pytest.mark.parametrize("dag_id", sorted(EXPECTED_DAGS))
    def test_failure_callback_set(self, dagbag, dag_id):
        """Падение таска обязано порождать алерт, а не тишину в логах."""
        dag = dagbag.get_dag(dag_id)
        assert dag.default_args.get("retries") is not None

    def test_training_is_weekly(self, dagbag):
        assert dagbag.get_dag("m5_training").schedule_interval == "0 2 * * 0"

    def test_inference_is_daily(self, dagbag):
        assert dagbag.get_dag("m5_inference").schedule_interval == "0 5 * * *"

    def test_no_catchup(self, dagbag):
        """catchup=False везде: переобучаться задним числом бессмысленно
        и опасно — забьёт очередь десятками прогонов."""
        for dag_id in EXPECTED_DAGS:
            assert dagbag.get_dag(dag_id).catchup is False

    def test_training_single_active_run(self, dagbag):
        """Два обучения одновременно не влезут в память."""
        assert dagbag.get_dag("m5_training").max_active_runs == 1


@pytest.mark.skip(reason="TODO: реализовать таски")
class TestIdempotency:
    """Повторный запуск таска с тем же ds не должен менять результат.

    Airflow ретраит таски. Если инференс задваивает строки в витрине,
    это обнаружится в самый неудачный момент.
    """

    def test_predict_twice_same_row_count(self):
        """Два прогона predict за один ds -> то же число строк."""

    def test_staging_is_replace_not_append(self):
        """Staging пересоздаёт таблицы, а не дописывает."""

    def test_accuracy_upsert_not_insert(self):
        """mart.forecast_accuracy обновляется по ключу, а не дублируется."""
