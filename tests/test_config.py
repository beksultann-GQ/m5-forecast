"""Тесты конфигов. Дешёвые, но ловят реальные грабли.

Например: кто-то поставил lag=7 в conf/features.yaml — и весь проект тихо
превращается в утечку. Тест ниже это ловит до первого обучения.
"""

from __future__ import annotations

import pytest

from m5.config import load_config
from m5.config.loader import config_hash, flatten


class TestConfigLoading:
    def test_loads_without_error(self, cfg):
        assert cfg.project.name == "m5-forecast"

    def test_defaults_are_merged(self, cfg):
        """Все файлы из defaults подмешались."""
        for section in ("data", "features", "model", "external", "validation"):
            assert section in cfg, f"секция {section} не подмешалась"

    def test_paths_are_absolute(self, cfg):
        """Пути абсолютные — иначе поведение зависит от cwd."""
        for key, value in cfg.paths.items():
            assert str(value).startswith("/"), f"{key} не абсолютный: {value}"

    def test_override_works(self):
        cfg = load_config(overrides=["model.params.learning_rate=0.5"])
        assert cfg.model.params.learning_rate == 0.5

    def test_flatten_produces_scalar_values(self, cfg):
        """MLflow.log_params не умеет вложенность — проверяем, что всё плоское."""
        flat = flatten(cfg)
        assert "model.params.learning_rate" in flat
        assert all(not isinstance(v, dict) for v in flat.values())

    def test_config_hash_is_stable(self, cfg):
        assert config_hash(cfg) == config_hash(cfg)
        assert len(config_hash(cfg)) == 12


class TestConfigInvariants:
    """Инварианты, нарушение которых ломает проект молча."""

    def test_all_lags_at_least_horizon(self, cfg):
        """Каждый лаг >= 28. Лаг меньше горизонта = утечка по построению.

        На инференсе продаж за последние 28 дней просто нет — их нечем заполнить.
        """
        horizon = cfg.project.horizon
        for lag in cfg.features.lags:
            assert lag >= horizon, f"lag={lag} < horizon={horizon}: фичу нечем посчитать в проде"

    def test_rolling_shift_equals_horizon(self, cfg):
        """Сдвиг скользящих окон равен горизонту."""
        assert cfg.features.rolling.shift == cfg.project.horizon

    def test_validation_gap_at_least_horizon(self, cfg):
        """gap между train и val не меньше горизонта."""
        assert cfg.validation.gap_days >= cfg.project.horizon

    def test_horizon_groups_cover_full_horizon(self, cfg):
        """Группы горизонтов покрывают 1..28 без дыр и пересечений."""
        covered: set[int] = set()
        for group in cfg.model.horizon_groups:
            days = set(range(group.start, group.end + 1))
            assert not (covered & days), f"группы пересекаются на {covered & days}"
            covered |= days
        assert covered == set(range(1, cfg.project.horizon + 1))

    def test_tweedie_variance_power_in_valid_range(self, cfg):
        """Tweedie осмысленен при 1 < p < 2 (смесь Пуассона и гаммы)."""
        p = cfg.model.params.tweedie_variance_power
        assert 1.0 < p < 2.0

    def test_weather_locations_cover_all_states(self, cfg):
        """Для каждого штата M5 задана точка — иначе часть магазинов
        останется без погодных фич, и это будет видно только по NULL'ам."""
        states = {loc.state_id for loc in cfg.external.weather.locations}
        assert states == {"CA", "TX", "WI"}

    def test_holiday_subdivisions_match_states(self, cfg):
        """Субдивизии Nager.Date соответствуют штатам M5."""
        assert set(cfg.external.holidays.subdivisions) == {"US-CA", "US-TX", "US-WI"}

    @pytest.mark.parametrize("layer", ["raw", "staging", "features", "mart"])
    def test_dwh_layers_defined(self, cfg, layer):
        assert layer in cfg.layers
