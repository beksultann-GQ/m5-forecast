-- Заглушка праздничных фич: выполняется, когда data/external/holidays.parquet нет.
-- См. комментарий в 02_ft_weather_stub.sql — состав колонок витрины
-- не должен зависеть от доступности внешнего источника.

CREATE SCHEMA IF NOT EXISTS ft;

CREATE OR REPLACE TABLE ft.holidays AS
SELECT
    CAST(NULL AS DATE)     AS date,
    CAST(NULL AS VARCHAR)  AS state_id,
    CAST(NULL AS TINYINT)  AS is_holiday,
    CAST(NULL AS BOOLEAN)  AS is_public_holiday,
    CAST(NULL AS VARCHAR)  AS holiday_name,
    CAST(NULL AS VARCHAR)  AS holiday_type,
    CAST(NULL AS INTEGER)  AS days_to_next_holiday,
    CAST(NULL AS INTEGER)  AS days_since_prev_holiday,
    CAST(NULL AS TINYINT)  AS is_day_before_holiday,
    CAST(NULL AS TINYINT)  AS is_day_after_holiday,
    CAST(NULL AS TINYINT)  AS is_pre_holiday_week,
    CAST(NULL AS TINYINT)  AS is_holiday_window
WHERE FALSE;
