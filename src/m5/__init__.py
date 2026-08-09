"""M5 Forecasting Accuracy — production-ready пайплайн.

Слои:
    data       — загрузка, конвертация в parquet, staging в DuckDB, валидация схем
    external   — внешние источники (погода Open-Meteo, праздники Nager.Date)
    features   — витрина фич на SQL (DuckDB), единый код для train и inference
    models     — baseline, обучение LightGBM, инференс, реестр моделей
    evaluation — WRMSSE/WMAPE, rolling-origin CV, бэктест
    monitoring — дрифт данных и прогнозов, фактическое качество
    serving    — FastAPI поверх витрины прогнозов
"""

__version__ = "0.1.0"
