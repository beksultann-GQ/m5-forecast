"""Тесты метрик. WRMSSE нетривиальна, и проверять её надо на синтетике
с ответом, посчитанным вручную, а не «выглядит правдоподобно».
"""

from __future__ import annotations

import numpy as np
import pytest

from m5.evaluation.metrics import bias, mae, rmse, rmsse, wmape


class TestSimpleMetrics:
    def test_mae_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_mae_known_value(self):
        assert mae(np.array([0.0, 0.0]), np.array([1.0, 3.0])) == 2.0

    def test_rmse_known_value(self):
        assert rmse(np.array([0.0, 0.0]), np.array([3.0, 4.0])) == pytest.approx(3.5355, rel=1e-3)

    def test_rmse_penalises_outliers_more_than_mae(self):
        """RMSE квадратичен — большая ошибка бьёт сильнее."""
        y_true = np.zeros(10)
        y_pred = np.array([0.0] * 9 + [10.0])
        assert rmse(y_true, y_pred) > mae(y_true, y_pred)

    def test_wmape_known_value(self):
        """sum|y-yhat| / sum|y| = 4/10 = 0.4"""
        y_true = np.array([4.0, 6.0])
        y_pred = np.array([2.0, 8.0])
        assert wmape(y_true, y_pred) == pytest.approx(0.4)

    def test_wmape_survives_zeros(self):
        """Главное свойство: нули в таргете не ломают метрику.

        В M5 ~68% значений нулевые — обычный MAPE здесь неприменим в принципе.
        """
        y_true = np.array([0.0, 0.0, 10.0])
        y_pred = np.array([1.0, 1.0, 10.0])
        assert wmape(y_true, y_pred) == pytest.approx(0.2)

    def test_wmape_all_zeros_is_nan(self):
        """Всё нулевое — метрика не определена, и это честнее, чем вернуть 0."""
        assert np.isnan(wmape(np.zeros(5), np.ones(5)))

    def test_bias_sign(self):
        """Перепрогноз даёт положительное смещение, недопрогноз — отрицательное."""
        y_true = np.array([10.0, 10.0])
        assert bias(y_true, np.array([12.0, 12.0])) > 0
        assert bias(y_true, np.array([8.0, 8.0])) < 0

    def test_rmsse_scaled_by_denominator(self):
        """RMSSE = RMSE / sqrt(scale)."""
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        assert rmsse(y_true, y_pred, scale=4.0) == pytest.approx(1.0)

    def test_rmsse_zero_scale_is_nan(self):
        """Константный ряд (scale=0) — RMSSE не определён, не делим на ноль."""
        assert np.isnan(rmsse(np.array([1.0]), np.array([1.0]), scale=0.0))


@pytest.mark.skip(reason="TODO: реализовать compute_weights и wrmsse")
class TestWRMSSE:
    """WRMSSE — метрика соревнования. 12 уровней агрегации, веса по выручке."""

    def test_weights_sum_to_one_within_level(self):
        """Внутри каждого уровня веса рядов суммируются в 1."""

    def test_weights_use_revenue_not_units(self):
        """Вес считается по выручке (sales × price), а не по штукам.

        Проверка на синтетике: два ряда с одинаковыми продажами в штуках,
        но разной ценой обязаны получить разные веса.
        """

    def test_scale_computed_on_train_only(self):
        """Знаменатель RMSSE считается по train, а не по val."""

    def test_scale_skips_leading_zeros(self):
        """Нули до первой продажи в знаменатель не входят.

        Иначе знаменатель у новых товаров занижен, и их вклад в метрику раздут.
        """

    def test_perfect_forecast_gives_zero(self):
        """Идеальный прогноз -> WRMSSE = 0."""

    def test_naive_forecast_gives_about_one(self):
        """Наивный прогноз даёт WRMSSE около 1 — метрика на него и нормирована.

        Это и есть смысл baseline: число ниже 1 значит «лучше наивного».
        """

    def test_twelve_levels_present(self):
        """Разбивка по уровням содержит ровно 12 записей."""

    def test_level_weights_are_equal(self):
        """Уровни усредняются с равным весом 1/12, а не по числу рядов."""


@pytest.mark.skip(reason="TODO: реализовать metrics_report")
class TestMetricsReport:
    def test_contains_all_configured_metrics(self):
        """В отчёте есть все метрики из conf/validation.yaml."""

    def test_horizon_breakdown_present(self):
        """Есть разбивка по группам горизонтов h1-7 ... h22-28."""

    def test_error_grows_with_horizon(self):
        """Ошибка обязана расти с горизонтом.

        Если не растёт — это почти всегда признак утечки, а не гениальной модели.
        """
