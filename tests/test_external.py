"""Тесты внешних источников. В сеть НЕ ходим — парсим зафиксированные ответы.

Правило: CI не должен зависеть от доступности Open-Meteo и Nager.Date.
Тесты, реально дёргающие API, помечены @pytest.mark.network и в CI не гоняются.
"""

from __future__ import annotations

from datetime import date

import pytest

from m5.external.geo import Location, subdivision_for_state
from m5.external.weather import build_archive_params, build_forecast_params


class TestGeo:
    def test_subdivision_mapping(self):
        assert subdivision_for_state("CA") == "US-CA"
        assert subdivision_for_state("TX") == "US-TX"
        assert subdivision_for_state("WI") == "US-WI"

    def test_unknown_state_raises(self):
        with pytest.raises(KeyError):
            subdivision_for_state("NY")

    def test_location_key_stable(self):
        """Ключ локации детерминирован — на нём строится кеш."""
        loc = Location("CA", "Los Angeles", 34.0522, -118.2437, "America/Los_Angeles")
        assert loc.key == "CA_34.0522_-118.2437"


class TestOpenMeteoParams:
    """Проверяем сборку запроса: контракт с API фиксируем тестом."""

    @pytest.fixture
    def loc(self) -> Location:
        return Location("CA", "Los Angeles", 34.0522, -118.2437, "America/Los_Angeles")

    def test_archive_params(self, loc: Location):
        params = build_archive_params(
            loc,
            start=date(2011, 1, 29),
            end=date(2016, 6, 19),
            daily_vars=["temperature_2m_max", "precipitation_sum"],
        )
        assert params["latitude"] == 34.0522
        assert params["start_date"] == "2011-01-29"
        assert params["end_date"] == "2016-06-19"
        assert params["daily"] == "temperature_2m_max,precipitation_sum"
        assert params["timezone"] == "America/Los_Angeles"

    def test_forecast_params_include_past_days(self, loc: Location):
        """past_days закрывает разрыв между archive (лаг ~5 дней) и сегодня."""
        params = build_forecast_params(loc, daily_vars=["temperature_2m_mean"])
        assert params["forecast_days"] == 16
        assert params["past_days"] == 7

    def test_units_are_passed_through(self, loc: Location):
        params = build_archive_params(
            loc,
            start=date(2015, 1, 1),
            end=date(2015, 1, 2),
            daily_vars=["temperature_2m_mean"],
            units={"temperature_unit": "celsius"},
        )
        assert params["temperature_unit"] == "celsius"


@pytest.mark.skip(reason="TODO: реализовать parse_open_meteo")
class TestWeatherParsing:
    def test_parse_produces_expected_columns(self, open_meteo_response):
        """Ответ API -> DataFrame со схемой WEATHER_SCHEMA."""

    def test_parse_adds_state_and_source(self, open_meteo_response):
        """В результат добавлены state_id и source ('archive'/'forecast')."""

    def test_missing_daily_key_raises(self):
        """Ответ без 'daily' — падаем явно, а не отдаём пустой фрейм."""

    def test_dates_are_continuous(self, open_meteo_response):
        """Нет дыр в датах — дыра означает потерянные строки при джойне."""


@pytest.mark.skip(reason="TODO: реализовать add_derived_features")
class TestWeatherDerived:
    def test_hdd_cdd_are_complementary(self):
        """hdd и cdd не бывают одновременно положительными."""

    def test_temp_anomaly_zero_on_average(self):
        """Аномалия относительно нормы в среднем по году около нуля."""

    def test_climatology_smoothed(self):
        """Норма по дню года сглажена окном ±7 дней — иначе она шумная."""

    def test_climatology_fills_far_horizon(self):
        """Дни 17-28 горизонта закрыты климатологией, source помечен."""


@pytest.mark.skip(reason="TODO: реализовать parse_holidays")
class TestHolidaysParsing:
    def test_global_holiday_applies_to_all_states(self, nager_response):
        """counties=null -> праздник во всех трёх штатах."""

    def test_regional_holiday_only_in_its_state(self, nager_response):
        """Cesar Chavez Day (counties=['US-CA']) есть только в CA.

        Ровно ради этого случая мы и берём Nager вместо event_name из M5:
        в M5 события плоские, без региональности.
        """

    def test_holiday_types_preserved(self, nager_response):
        """Public / Bank / School различаются — эффект на трафик разный."""

    def test_day_before_holiday_flag(self):
        """is_day_before_holiday=1 ровно за день до праздника.

        Главный рабочий сигнал: закупаются накануне, а не в сам праздник.
        """

    def test_days_to_next_holiday_monotonic(self):
        """Счётчик до праздника убывает на 1 в день и обнуляется в праздник."""


@pytest.mark.skip(reason="TODO: реализовать CachedJSONClient")
class TestCachedClient:
    def test_cache_key_is_stable(self):
        """Один и тот же запрос -> один и тот же ключ независимо от порядка params."""

    def test_second_call_hits_cache(self, tmp_path):
        """Повторный запрос не идёт в сеть."""

    def test_offline_mode_raises_on_miss(self, tmp_path):
        """В offline промах кеша — явная ошибка, а не тихий поход в сеть.

        Нужно для CI: пайплайн не должен молча зависеть от доступности API.
        """

    def test_retries_on_5xx(self):
        """5xx ретраится с экспоненциальной паузой."""

    def test_no_retry_on_4xx(self):
        """400/404 не ретраятся — это ошибка запроса, а не сети."""


@pytest.mark.network
@pytest.mark.skip(reason="TODO: включить после реализации клиента")
class TestLiveAPIs:
    """Реальные запросы. Не гоняются в CI, запускаются руками: pytest -m network."""

    def test_open_meteo_archive_responds(self):
        """Контракт Open-Meteo не изменился."""

    def test_nager_date_responds(self):
        """Контракт Nager.Date не изменился."""
