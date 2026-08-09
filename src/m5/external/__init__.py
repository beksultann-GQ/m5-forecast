"""Внешние источники данных сверх Kaggle-датасета.

Гипотезы, которые проверяем:
    weather  — погода двигает продажи еды и сезонных товаров
               (жара -> напитки, снег -> провал трафика в магазин)
    holidays — независимый календарь госпраздников США по штатам,
               в M5 event_name_1/2 покрывает не всё и без региональности

Оба источника бесплатные и без ключа (Open-Meteo Archive, Nager.Date).
"""

from m5.external.holidays import fetch_holidays
from m5.external.weather import fetch_weather

__all__ = ["fetch_holidays", "fetch_weather"]
