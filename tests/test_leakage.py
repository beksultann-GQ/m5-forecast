"""Тесты на утечку. Самые важные тесты в проекте.

Утечка через лаги — ошибка №1 в прогнозировании временных рядов:
скор на валидации великолепный, на тесте катастрофа. Причём молча —
ничего не падает, просто модель в проде не работает.

Тест на утечку в pet-проекте — сильный сигнал на собеседовании.
Он показывает, что человек понимает, где именно всё ломается.
"""

from __future__ import annotations

from datetime import date, timedelta
from itertools import pairwise

import pytest

from m5.evaluation.cv import Fold, RollingOriginSplitter, assert_fold_integrity


class TestFoldIntegrity:
    """Границы фолдов не должны пересекаться. Это проверяется по построению."""

    def test_train_end_strictly_before_val_start(self):
        """train не имеет права заходить в val даже на день."""
        with pytest.raises(ValueError, match="утечка"):
            Fold(
                index=0,
                train_start=date(2015, 1, 1),
                train_end=date(2016, 1, 31),
                val_start=date(2016, 1, 31),  # тот же день — уже утечка
                val_end=date(2016, 2, 27),
            )

    def test_gap_equals_horizon(self):
        """Между train и val ровно 28 дней разрыва."""
        fold = Fold(
            index=0,
            train_start=date(2015, 1, 1),
            train_end=date(2016, 1, 1),
            val_start=date(2016, 1, 30),  # +29 дней => gap 28
            val_end=date(2016, 2, 26),
        )
        assert fold.gap_days == 28
        assert fold.horizon_days == 28
        assert_fold_integrity(fold, horizon=28)

    def test_gap_smaller_than_horizon_rejected(self):
        """gap < horizon — лаги протекут. Такой сплиттер создать нельзя."""
        with pytest.raises(ValueError, match="протекут"):
            RollingOriginSplitter(horizon=28, gap_days=7)

    def test_short_gap_fold_fails_integrity(self):
        """Даже если фолд собрали руками, проверка обязана его отклонить."""
        fold = Fold(
            index=0,
            train_start=date(2015, 1, 1),
            train_end=date(2016, 1, 1),
            val_start=date(2016, 1, 8),  # gap всего 6 дней
            val_end=date(2016, 2, 4),
        )
        with pytest.raises(ValueError, match="gap"):
            assert_fold_integrity(fold, horizon=28)


class TestRollingOriginSplit:
    """Нарезка фолдов. Ошибка здесь отравляет все метрики проекта разом."""

    @pytest.fixture
    def folds(self):
        splitter = RollingOriginSplitter(n_folds=4, min_train_days=365)
        return splitter.split(date(2013, 1, 1), date(2016, 6, 19))

    def test_produces_requested_number_of_folds(self, folds):
        assert len(folds) == 4

    def test_folds_do_not_overlap(self, folds):
        """Валидационные окна разных фолдов не пересекаются."""
        windows = sorted((f.val_start, f.val_end) for f in folds)
        for (_, prev_end), (next_start, _) in pairwise(windows):
            assert prev_end < next_start

    def test_folds_go_backwards_in_time(self, folds):
        """Фолд 0 — самый свежий, дальше лесенкой назад."""
        assert folds[0].val_end > folds[-1].val_end

    def test_first_fold_ends_at_data_end(self, folds):
        """Фолд 0 заканчивается на последней доступной дате: свежий отрезок
        обязан участвовать в оценке, иначе метрика описывает прошлогоднюю модель."""
        assert folds[0].val_end == date(2016, 6, 19)

    def test_every_fold_passes_integrity(self, folds):
        for fold in folds:
            assert_fold_integrity(fold, horizon=28)

    def test_gap_holds_on_every_fold(self, folds):
        assert all(fold.gap_days == 28 for fold in folds)

    def test_forecast_origin_is_day_before_validation(self, folds):
        """origin — это день перед val, а не train_end.

        train_end отделён gap'ом (там мы не учимся), но прогнозируем мы
        из последнего известного дня. Путаница этих двух дат сдвигает
        весь горизонт на 28 дней и делает метрики бессмысленными.
        """
        for fold in folds:
            assert fold.forecast_origin == fold.val_start - timedelta(days=1)
            assert fold.forecast_origin > fold.train_end

    def test_expanding_window_shares_train_start(self, folds):
        """При expanding=True начало обучения одно на все фолды."""
        assert len({fold.train_start for fold in folds}) == 1

    def test_raises_when_history_too_short(self):
        """Не хватает истории на n_folds — падаем с понятным сообщением,
        а не отдаём молча меньше фолдов."""
        splitter = RollingOriginSplitter(n_folds=4, min_train_days=730)
        with pytest.raises(ValueError, match="хватает не на все"):
            splitter.split(date(2015, 1, 1), date(2016, 6, 19))


@pytest.mark.skip(reason="TODO: реализовать build_features")
class TestFeatureLeakage:
    """Главный класс тестов: фичи не смотрят в будущее."""

    def test_no_dates_after_as_of(self):
        """В витрине нет ни одной строки с таргетом позже as_of."""

    def test_min_lag_is_at_least_horizon(self):
        """Все лаги >= 28. lag_7 в витрине быть не должно физически."""

    def test_rolling_windows_are_shifted(self):
        """roll_mean_7 на дату T равен среднему продаж за [T-34, T-28].

        Проверяется на синтетике с известным рядом: считаем ожидаемое значение
        руками и сравниваем. Именно этот тест ловит забытый shift.
        """

    def test_rolling_excludes_current_day(self):
        """Продажи дня T не входят ни в одну фичу дня T."""

    def test_feature_value_stable_when_future_changes(self):
        """Ключевая проверка: меняем продажи ПОСЛЕ as_of — фичи до as_of
        обязаны остаться байт-в-байт теми же.

        Если хоть одна фича изменилась — она смотрит в будущее.
        Это самый надёжный способ поймать утечку, потому что он не требует
        знать, какая именно фича виновата.
        """

    def test_price_features_use_only_past_weeks(self):
        """Ценовые фичи не используют цены будущих недель."""


@pytest.mark.skip(reason="TODO: реализовать build_features(mode='inference')")
class TestTrainServingConsistency:
    """Одна функция считает фичи для train и для inference."""

    def test_same_function_same_result(self):
        """Фичи на пересечении дат train и inference совпадают."""

    def test_feature_columns_identical(self):
        """Состав и порядок колонок идентичны в обоих режимах."""

    def test_categorical_codes_match(self):
        """Коды категорий на инференсе те же, что зафиксированы на train."""


@pytest.mark.skip(reason="TODO: реализовать external.weather")
class TestExternalDataLeakage:
    """Внешние данные — отдельный риск утечки."""

    def test_weather_source_marked_on_future_dates(self):
        """На датах после as_of weather_source != 'archive'.

        Если на будущих датах оказался archive — значит, в train попала
        фактическая погода, которой в проде не будет. Это тихая утечка,
        которая на бэктесте выглядит как улучшение качества.
        """

    def test_holidays_available_for_future(self):
        """Праздники, наоборот, обязаны быть известны на весь горизонт.

        Это их главное преимущество перед погодой: никакого skew.
        """
