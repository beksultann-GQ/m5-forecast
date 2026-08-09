"""Тесты фич. Проверяем на синтетике, где ответ известен заранее.

Отдельно от test_leakage.py: здесь про корректность вычислений,
там — про то, что фичи не смотрят в будущее.
"""

from __future__ import annotations

import pytest

from m5.data.duck import render_sql


class TestSQLRendering:
    def test_placeholders_substituted(self):
        sql = "SELECT * FROM read_parquet('{{ raw_dir }}/calendar.parquet')"
        out = render_sql(sql, {"raw_dir": "/data/raw"})
        assert out == "SELECT * FROM read_parquet('/data/raw/calendar.parquet')"

    def test_multiple_placeholders(self):
        sql = "WHERE date <= '{{ as_of }}' AND {{ store_filter }}"
        out = render_sql(sql, {"as_of": "2016-04-24", "store_filter": "TRUE"})
        assert "2016-04-24" in out
        assert "TRUE" in out

    def test_missing_placeholder_raises(self):
        """Незаполненный плейсхолдер — явная ошибка, а не битый SQL в проде."""
        with pytest.raises(KeyError, match="as_of"):
            render_sql("WHERE date <= '{{ as_of }}'", {})

    def test_whitespace_tolerant(self):
        assert render_sql("{{raw_dir}}", {"raw_dir": "x"}) == "x"
        assert render_sql("{{  raw_dir  }}", {"raw_dir": "x"}) == "x"


@pytest.mark.skip(reason="TODO: реализовать staging SQL")
class TestStaging:
    def test_unpivot_row_count(self, duckdb_con, synthetic_sales):
        """После разворота строк ровно n_series × n_days."""

    def test_wide_to_long_preserves_totals(self, duckdb_con, synthetic_sales):
        """Сумма продаж до и после разворота совпадает."""

    def test_first_sale_date_correct(self, duckdb_con, synthetic_sales):
        """first_sale_date = первый день с sales > 0.

        В синтетике половина рядов стартует на 100-й день — проверяем именно их.
        """

    def test_pre_release_rows_dropped(self, duckdb_con, synthetic_sales):
        """Нули до первой продажи выброшены из витрины.

        Это не нулевой спрос, а отсутствие товара. Учиться на них — учиться на мусоре.
        """

    def test_prices_joined_by_week(self, duckdb_con, synthetic_prices):
        """Недельная цена корректно размножилась на дни недели."""

    def test_missing_price_marks_out_of_assortment(self, duckdb_con):
        """Нет цены на неделю -> is_out_of_assortment = 1."""


@pytest.mark.skip(reason="TODO: реализовать ft-слой")
class TestLagFeatures:
    def test_lag_28_shifts_exactly_28_days(self, duckdb_con, synthetic_sales):
        """sales_lag_28 на дату T равен sales на дату T-28. Точно, не примерно."""

    def test_lag_null_at_series_start(self, duckdb_con, synthetic_sales):
        """В начале ряда лаг NULL, а не ноль. Ноль — это «продали 0 штук»."""

    def test_lag_364_preserves_day_of_week(self, duckdb_con, synthetic_sales):
        """364 = 52×7, день недели сохраняется."""

    def test_lags_partitioned_by_series(self, duckdb_con, synthetic_sales):
        """Лаг не перетекает между рядами: конец одного ряда не попадает в начало другого."""


@pytest.mark.skip(reason="TODO: реализовать ft-слой")
class TestRollingFeatures:
    def test_roll_mean_7_matches_manual(self, duckdb_con, synthetic_sales):
        """roll_mean_7(T) == mean(sales[T-34 .. T-28]). Считаем руками и сравниваем."""

    def test_zero_ratio_matches_manual(self, duckdb_con, synthetic_sales):
        """zero_ratio_28 == доля нулей в окне."""

    def test_dow_mean_uses_same_weekday(self, duckdb_con, synthetic_sales):
        """dow_mean_8w усредняет только тот же день недели."""

    def test_dept_aggregate_equals_sum_of_items(self, duckdb_con, synthetic_sales):
        """Агрегат отдела равен сумме товаров этого отдела в магазине."""


@pytest.mark.skip(reason="TODO: реализовать feature_columns")
class TestFeatureContract:
    def test_column_order_deterministic(self, cfg):
        """Порядок колонок стабилен между вызовами.

        LightGBM работает с позициями, а не с именами: разъехавшийся порядок
        не вызовет ошибку, но предсказания будут мусором.
        """

    def test_all_configured_features_present(self, cfg):
        """Всё, что перечислено в conf/features.yaml, есть в витрине."""

    def test_no_target_leak_in_feature_list(self, cfg):
        """В списке фич нет самого таргета и его производных без сдвига."""

    def test_categorical_columns_declared(self, cfg):
        """Все категориальные колонки объявлены — иначе LightGBM
        воспримет их коды как числа с порядком."""
