-- Фичи: погода (Open-Meteo). Джойн по (state_id, date).
--
-- Источник: data/external/weather.parquet, собирается m5.external.weather
--
-- ЧТО ЗДЕСЬ ВАЖНО ПРО УТЕЧКУ:
--   Погода на день T известна только ПОСЛЕ дня T. На train мы её знаем,
--   на inference за горизонт 28 дней — нет. Поэтому колонка `source`
--   обязана доезжать до витрины: она показывает, факт это (archive),
--   прогноз (forecast) или климатическая норма (climatology).
--   На бэктесте надо сравнить качество с погодой и без — если прироста нет,
--   фичу выкидываем, а не тащим в прод «потому что данных больше».

CREATE SCHEMA IF NOT EXISTS ft;

CREATE OR REPLACE TABLE ft.weather AS
WITH src AS (
    SELECT * FROM read_parquet('{{ external_dir }}/weather.parquet')
),
norms AS (
    -- климатическая норма по дню года: базис для аномалий
    SELECT
        state_id,
        DAYOFYEAR(date) AS dayofyear,
        AVG(temperature_2m_mean) AS temp_norm,
        AVG(precipitation_sum)   AS precip_norm
    FROM src
    WHERE source = 'archive'
    GROUP BY 1, 2
),
pctl AS (
    SELECT
        state_id,
        QUANTILE_CONT(precipitation_sum, 0.90) AS precip_p90
    FROM src
    WHERE source = 'archive'
    GROUP BY 1
)
SELECT
    s.state_id,
    s.date,
    s.source                                   AS weather_source,

    s.temperature_2m_max,
    s.temperature_2m_min,
    s.temperature_2m_mean,
    s.apparent_temperature_max,
    s.precipitation_sum,
    s.snowfall_sum,
    s.windspeed_10m_max,
    s.shortwave_radiation_sum,

    -- отклонение от нормы: сама температура без контекста бесполезна,
    -- +25C в январе в WI и +25C в июле в CA значат совершенно разное
    s.temperature_2m_mean - n.temp_norm        AS temp_anomaly,
    s.precipitation_sum   - n.precip_norm      AS precip_anomaly,

    -- градусо-дни: нелинейный отклик спроса на температуру
    GREATEST(18.0 - s.temperature_2m_mean, 0)  AS hdd,   -- отопление
    GREATEST(s.temperature_2m_mean - 18.0, 0)  AS cdd,   -- охлаждение

    CASE WHEN s.precipitation_sum > p.precip_p90 THEN 1 ELSE 0 END AS is_heavy_rain,
    CASE WHEN s.snowfall_sum > 0 THEN 1 ELSE 0 END                 AS is_snow_day,
    CASE WHEN s.temperature_2m_max > 30 THEN 1 ELSE 0 END          AS is_hot_day,
    CASE WHEN s.temperature_2m_min < 0  THEN 1 ELSE 0 END          AS is_freezing_day,

    -- резкие перепады: люди реагируют на смену погоды сильнее, чем на её уровень
    s.temperature_2m_mean - LAG(s.temperature_2m_mean) OVER w      AS temp_change_1d,
    AVG(s.temperature_2m_mean) OVER (w ROWS BETWEEN 6 PRECEDING AND 0 PRECEDING)
        AS temp_mean_7d

FROM src s
LEFT JOIN norms n ON n.state_id = s.state_id AND n.dayofyear = DAYOFYEAR(s.date)
LEFT JOIN pctl  p ON p.state_id = s.state_id
WINDOW w AS (PARTITION BY s.state_id ORDER BY s.date);
