-- Фичи: госпраздники (Nager.Date). Джойн по (state_id, date).
--
-- Источник: data/external/holidays.parquet, собирается m5.external.holidays
--
-- В отличие от погоды, утечки здесь нет вообще: календарь праздников
-- известен на годы вперёд и одинаково доступен на train и в проде.
-- Это «бесплатная» фича в терминах training/serving skew.
--
-- Смысл не в самом дне праздника, а в его окрестности: закупаются НАКАНУНЕ,
-- в сам праздник магазин полупустой или закрыт.

CREATE SCHEMA IF NOT EXISTS ft;

CREATE OR REPLACE TABLE ft.holidays AS
WITH src AS (
    SELECT
        CAST(date AS DATE)                     AS date,
        subdivision,
        REPLACE(subdivision, 'US-', '')        AS state_id,
        holiday_name,
        holiday_type,
        is_public_holiday
    FROM read_parquet('{{ external_dir }}/holidays.parquet')
),
-- полная сетка (дата × штат): без неё дни без праздника выпадут из витрины
grid AS (
    SELECT c.date, s.state_id
    FROM (SELECT DISTINCT date FROM stg.calendar) c
    CROSS JOIN (SELECT DISTINCT state_id FROM stg.hierarchy) s
),
joined AS (
    SELECT
        g.date,
        g.state_id,
        h.holiday_name,
        h.holiday_type,
        COALESCE(h.is_public_holiday, FALSE)                     AS is_public_holiday,
        CASE WHEN h.holiday_name IS NOT NULL THEN 1 ELSE 0 END   AS is_holiday
    FROM grid g
    LEFT JOIN src h ON h.date = g.date AND h.state_id = g.state_id
),
-- расстояния до ближайшего праздника в обе стороны
distances AS (
    SELECT
        j.*,
        -- ближайший будущий праздник в этом штате
        DATEDIFF('day', j.date, MIN(nxt.date)) AS days_to_next_holiday,
        DATEDIFF('day', MAX(prv.date), j.date) AS days_since_prev_holiday
    FROM joined j
    LEFT JOIN joined nxt
           ON nxt.state_id = j.state_id AND nxt.is_holiday = 1 AND nxt.date >= j.date
    LEFT JOIN joined prv
           ON prv.state_id = j.state_id AND prv.is_holiday = 1 AND prv.date <= j.date
    GROUP BY ALL
)
SELECT
    date,
    state_id,
    is_holiday,
    is_public_holiday,
    COALESCE(holiday_name, 'none')                     AS holiday_name,
    COALESCE(holiday_type, 'none')                     AS holiday_type,
    days_to_next_holiday,
    days_since_prev_holiday,

    -- окрестность праздника: главный рабочий сигнал
    CASE WHEN days_to_next_holiday = 1 THEN 1 ELSE 0 END        AS is_day_before_holiday,
    CASE WHEN days_since_prev_holiday = 1 THEN 1 ELSE 0 END     AS is_day_after_holiday,
    CASE WHEN days_to_next_holiday <= 3 THEN 1 ELSE 0 END       AS is_pre_holiday_week,
    CASE WHEN days_to_next_holiday <= 3
           OR days_since_prev_holiday <= 3 THEN 1 ELSE 0 END    AS is_holiday_window

FROM distances;
