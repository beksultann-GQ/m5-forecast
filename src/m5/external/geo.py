"""Привязка магазинов M5 к географии.

В датасете есть только store_id (CA_1..CA_4, TX_1..TX_3, WI_1..WI_3) и state_id.
Точных адресов Walmart не публикует, поэтому берём по одной репрезентативной
точке на штат — гранулярнее погоды нам всё равно не получить честно.

Это осознанное упрощение, и его надо проговаривать: внутри штата
погода в разных городах различается, а мы этим различием пренебрегаем.
"""

from __future__ import annotations

from dataclasses import dataclass

from m5.config import Config


@dataclass(frozen=True)
class Location:
    """Точка, для которой тянем погоду."""

    state_id: str
    city: str
    lat: float
    lon: float
    tz: str

    @property
    def key(self) -> str:
        return f"{self.state_id}_{self.lat:.4f}_{self.lon:.4f}"


# Штаты M5 и их ISO-3166-2 коды для Nager.Date
STATE_TO_SUBDIVISION = {
    "CA": "US-CA",
    "TX": "US-TX",
    "WI": "US-WI",
}

STORE_TO_STATE = {
    "CA_1": "CA",
    "CA_2": "CA",
    "CA_3": "CA",
    "CA_4": "CA",
    "TX_1": "TX",
    "TX_2": "TX",
    "TX_3": "TX",
    "WI_1": "WI",
    "WI_2": "WI",
    "WI_3": "WI",
}


def locations_from_config(cfg: Config) -> list[Location]:
    """Собрать список точек из conf/external.yaml."""
    return [Location(**dict(loc)) for loc in cfg.external.weather.locations]


def location_for_state(cfg: Config, state_id: str) -> Location:
    """Точка по коду штата."""
    for loc in locations_from_config(cfg):
        if loc.state_id == state_id:
            return loc
    raise KeyError(f"Нет координат для штата {state_id} в conf/external.yaml")


def subdivision_for_state(state_id: str) -> str:
    """'CA' -> 'US-CA' для Nager.Date."""
    try:
        return STATE_TO_SUBDIVISION[state_id]
    except KeyError as exc:
        raise KeyError(f"Неизвестный штат: {state_id}") from exc
