-- Заглушка погодных фич: выполняется, когда data/external/weather.parquet нет.
--
-- Зачем пустая таблица вместо «убрать джойн»: витрина ft.mart должна иметь
-- ОДИН И ТОТ ЖЕ состав колонок независимо от того, доехали внешние данные или нет.
-- Иначе модель, обученная с погодой, на инференсе без погоды упадёт на несовпадении
-- фич — либо, что хуже, молча предскажет мусор.
--
-- Здесь колонки есть, значения NULL. LightGBM умеет работать с NULL нативно.

CREATE SCHEMA IF NOT EXISTS ft;

CREATE OR REPLACE TABLE ft.weather AS
SELECT
    CAST(NULL AS VARCHAR) AS state_id,
    CAST(NULL AS DATE)    AS date,
    CAST(NULL AS VARCHAR) AS weather_source,
    CAST(NULL AS FLOAT)   AS temperature_2m_max,
    CAST(NULL AS FLOAT)   AS temperature_2m_min,
    CAST(NULL AS FLOAT)   AS temperature_2m_mean,
    CAST(NULL AS FLOAT)   AS apparent_temperature_max,
    CAST(NULL AS FLOAT)   AS precipitation_sum,
    CAST(NULL AS FLOAT)   AS snowfall_sum,
    CAST(NULL AS FLOAT)   AS windspeed_10m_max,
    CAST(NULL AS FLOAT)   AS shortwave_radiation_sum,
    CAST(NULL AS FLOAT)   AS temp_anomaly,
    CAST(NULL AS FLOAT)   AS precip_anomaly,
    CAST(NULL AS FLOAT)   AS hdd,
    CAST(NULL AS FLOAT)   AS cdd,
    CAST(NULL AS TINYINT) AS is_heavy_rain,
    CAST(NULL AS TINYINT) AS is_snow_day,
    CAST(NULL AS TINYINT) AS is_hot_day,
    CAST(NULL AS TINYINT) AS is_freezing_day,
    CAST(NULL AS FLOAT)   AS temp_change_1d,
    CAST(NULL AS FLOAT)   AS temp_mean_7d
WHERE FALSE;
