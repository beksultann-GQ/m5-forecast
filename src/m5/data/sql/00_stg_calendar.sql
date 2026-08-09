-- Staging: календарь.
-- Источник: data/raw/calendar.parquet (1969 строк, d_1 .. d_1969)
-- Здесь же нормализуем SNAP: в raw это три колонки snap_CA/TX/WI,
-- в модель удобнее один флаг, выбранный по штату магазина.

CREATE SCHEMA IF NOT EXISTS stg;

CREATE OR REPLACE TABLE stg.calendar AS
WITH src AS (
    SELECT * FROM read_parquet('{{ raw_dir }}/calendar.parquet')
)
SELECT
    CAST(date AS DATE)                                   AS date,
    d,
    CAST(REPLACE(d, 'd_', '') AS INTEGER)                AS day_index,
    wm_yr_wk,
    wday,
    weekday,
    month,
    year,

    -- календарные признаки
    DAYOFWEEK(CAST(date AS DATE))                        AS dayofweek,
    DAYOFMONTH(CAST(date AS DATE))                       AS dayofmonth,
    WEEKOFYEAR(CAST(date AS DATE))                       AS weekofyear,
    DAYOFYEAR(CAST(date AS DATE))                        AS dayofyear,
    QUARTER(CAST(date AS DATE))                          AS quarter,
    CASE WHEN wday IN (1, 2) THEN 1 ELSE 0 END           AS is_weekend,  -- в M5 wday: 1=Sat, 2=Sun
    CASE WHEN DAYOFMONTH(CAST(date AS DATE)) <= 7  THEN 1 ELSE 0 END AS is_month_start,
    CASE WHEN DAYOFMONTH(CAST(date AS DATE)) >= 24 THEN 1 ELSE 0 END AS is_month_end,

    -- события M5 (NULL там, где события нет)
    NULLIF(event_name_1, '')                             AS event_name_1,
    NULLIF(event_type_1, '')                             AS event_type_1,
    NULLIF(event_name_2, '')                             AS event_name_2,
    NULLIF(event_type_2, '')                             AS event_type_2,
    CASE WHEN NULLIF(event_name_1, '') IS NOT NULL
           OR NULLIF(event_name_2, '') IS NOT NULL
         THEN 1 ELSE 0 END                               AS has_event,

    snap_CA,
    snap_TX,
    snap_WI
FROM src;


-- Длинная форма SNAP: (date, state_id, snap) — джойнится к продажам по state_id
CREATE OR REPLACE TABLE stg.snap AS
SELECT date, 'CA' AS state_id, snap_CA AS snap FROM stg.calendar
UNION ALL
SELECT date, 'TX' AS state_id, snap_TX AS snap FROM stg.calendar
UNION ALL
SELECT date, 'WI' AS state_id, snap_WI AS snap FROM stg.calendar;


-- Расстояние до ближайшего события в обе стороны.
-- Считается один раз по календарю, поэтому утечки нет: календарь известен наперёд.
CREATE OR REPLACE TABLE stg.calendar_events AS
WITH event_days AS (
    SELECT date FROM stg.calendar WHERE has_event = 1
)
SELECT
    c.date,
    DATEDIFF('day', c.date, (SELECT MIN(e.date) FROM event_days e WHERE e.date >= c.date))
        AS days_to_next_event,
    DATEDIFF('day', (SELECT MAX(e.date) FROM event_days e WHERE e.date <= c.date), c.date)
        AS days_since_prev_event
FROM stg.calendar c;
