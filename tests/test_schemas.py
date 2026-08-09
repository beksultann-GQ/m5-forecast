"""Тесты валидации данных. Пайплайн обязан ПАДАТЬ на битых данных.

Проверяем не то, что валидация пропускает хорошее (это скучно),
а то, что она ловит плохое. Валидация, которая ничего не ловит, хуже отсутствия
валидации: она создаёт ложное ощущение защищённости.
"""

from __future__ import annotations

import pytest

from m5.data.schemas import SCHEMAS


class TestSchemaDefinitions:
    def test_all_schemas_registered(self):
        expected = {
            "calendar",
            "sell_prices",
            "sales_long",
            "features",
            "predictions",
            "weather",
            "holidays",
        }
        assert expected <= set(SCHEMAS)

    def test_sales_long_unique_key(self):
        """Ключ (id, date) уникален — дубли задваивают продажи."""
        assert SCHEMAS["sales_long"].unique == ["id", "date"]

    def test_predictions_unique_key(self):
        assert SCHEMAS["predictions"].unique == ["id", "date"]

    def test_weather_unique_key(self):
        """Дубль по (state_id, date) при джойне размножит строки продаж."""
        assert SCHEMAS["weather"].unique == ["state_id", "date"]


@pytest.mark.skip(reason="TODO: реализовать validate()")
class TestValidationCatchesBadData:
    def test_negative_sales_rejected(self, synthetic_sales):
        """Отрицательные продажи -> ошибка."""

    def test_duplicate_id_date_rejected(self, synthetic_sales):
        """Дубли по (id, date) -> ошибка."""

    def test_zero_price_rejected(self, synthetic_prices):
        """Цена 0 или отрицательная -> ошибка. Ноль ломает все относительные фичи."""

    def test_unknown_state_rejected(self, synthetic_sales):
        """state_id вне {CA, TX, WI} -> ошибка."""

    def test_date_gap_detected(self, synthetic_calendar):
        """Дыра в календаре -> ошибка.

        Пропущенный день молча сдвигает все лаги на этом ряду.
        """

    def test_nan_in_predictions_rejected(self):
        """NaN в прогнозе -> ошибка. В витрину такое пускать нельзя."""

    def test_negative_prediction_rejected(self):
        """Отрицательный прогноз продаж физически невозможен."""

    def test_impossible_temperature_rejected(self):
        """Температура +200C -> внешний API отдал мусор, надо падать."""


@pytest.mark.skip(reason="TODO: реализовать sql_checks()")
class TestSQLChecks:
    def test_returns_zero_violations_on_clean_data(self, duckdb_con, synthetic_sales):
        """На чистых данных все счётчики нарушений = 0."""

    def test_detects_duplicates(self, duckdb_con):
        """Дубль в таблице обнаружен без вытаскивания её в память."""

    def test_detects_future_dates(self, duckdb_con):
        """Даты позже as_of обнаружены — это индикатор утечки."""

    def test_detects_missing_prices(self, duckdb_con):
        """Товар продаётся, но цены нет — дыра в справочнике."""


@pytest.mark.skip(reason="TODO: реализовать sanity_checks()")
class TestForecastSanityChecks:
    """Гейт перед публикацией прогноза в витрину."""

    def test_row_count_equals_series_times_horizon(self):
        """Строк ровно n_series × 28 — ничего не потерялось при джойнах."""

    def test_all_horizons_present(self):
        """Присутствуют все горизонты 1..28."""

    def test_volume_within_bounds(self):
        """Суммарный объём в пределах ±40% от факта за прошлые 28 дней.

        Ловит катастрофические сбои: не ту модель подгрузили,
        фичи собрались из пустой витрины и так далее.
        """

    def test_fails_closed(self):
        """При провале любой проверки в витрину НЕ пишем.

        Плохой прогноз хуже отсутствия прогноза: по нему закупят товар.
        """
