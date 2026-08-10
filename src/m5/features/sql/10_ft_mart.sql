-- Витрина фич: всё вместе. Это единственная таблица, которую читают
-- train, backtest и inference. Один SQL для всех трёх — так не бывает skew.
--
-- Параметры:
--   {{ as_of }}   — граница знания. Ничего позже этой даты в фичи не попадает.
--   {{ horizon }} — 28
--   {{ external_dir }}
--   {{ store_filter }} — 'TRUE' либо "store_id IN ('CA_1','CA_2')"

CREATE SCHEMA IF NOT EXISTS ft;

CREATE OR REPLACE TABLE ft.mart AS
SELECT
    -- ключи
    e.id,
    e.date,
    e.item_id,
    e.dept_id,
    e.cat_id,
    e.store_id,
    e.state_id,

    -- таргет (NULL на inference)
    e.sales,

    -- лаги
    l.sales_lag_28,
    l.sales_lag_35,
    l.sales_lag_42,
    l.sales_lag_56,
    l.sales_lag_364,
    l.sales_diff_28_35,
    l.sales_diff_28_56,
    l.sales_same_dow_mean_3w,

    -- скользящие статистики (все сдвинуты на горизонт)
    r.roll_mean_7,
    r.roll_std_7,
    r.roll_max_7,
    r.roll_mean_28,
    r.roll_std_28,
    r.roll_max_28,
    r.roll_mean_180,
    r.roll_std_180,
    r.zero_ratio_28,
    r.zero_ratio_180,
    r.nonzero_days_28,
    r.dow_mean_8w,
    r.trend_28_over_180,

    -- агрегаты уровнем выше
    ra.dept_roll_mean_28,
    ra.cat_roll_mean_28,

    -- календарь
    e.dayofweek,
    e.dayofmonth,
    e.weekofyear,
    e.dayofyear,
    e.month,
    e.year,
    e.quarter,
    e.is_weekend,
    e.is_month_start,
    e.is_month_end,
    e.event_name_1,
    e.event_type_1,
    e.event_name_2,
    e.event_type_2,
    e.days_to_next_event,
    e.days_since_prev_event,
    e.snap,

    -- цены
    e.sell_price,
    e.price_change_1w,
    e.price_momentum_4w,
    e.is_price_drop,
    e.price_rel_own_mean,
    e.price_rel_own_max,
    e.price_std_item_store,
    e.price_nunique_item,
    e.price_rel_item_mean,
    e.price_rel_dept_mean,
    e.price_rel_cat_mean,

    -- внешние: погода
    w.temperature_2m_mean,
    w.temperature_2m_max,
    w.temperature_2m_min,
    w.precipitation_sum,
    w.snowfall_sum,
    w.windspeed_10m_max,
    w.temp_anomaly,
    w.precip_anomaly,
    w.hdd,
    w.cdd,
    w.is_heavy_rain,
    w.is_snow_day,
    w.is_hot_day,
    w.is_freezing_day,
    w.temp_change_1d,
    w.temp_mean_7d,
    w.weather_source,       -- диагностика: archive / forecast / climatology

    -- внешние: праздники
    h.is_holiday,
    h.is_public_holiday,
    h.holiday_name,
    h.holiday_type,
    h.days_to_next_holiday,
    h.days_since_prev_holiday,
    h.is_day_before_holiday,
    h.is_day_after_holiday,
    h.is_pre_holiday_week,
    h.is_holiday_window,

    -- служебное
    e.days_since_first_sale,
    e.is_out_of_assortment

FROM stg.sales_enriched e
LEFT JOIN ft.lags        l  ON l.id = e.id  AND l.date = e.date
LEFT JOIN ft.rolling     r  ON r.id = e.id  AND r.date = e.date
LEFT JOIN ft.rolling_agg ra ON ra.store_id = e.store_id
                           AND ra.dept_id  = e.dept_id
                           AND ra.date     = e.date
LEFT JOIN ft.weather     w  ON w.state_id = e.state_id AND w.date = e.date
LEFT JOIN ft.holidays    h  ON h.state_id = e.state_id AND h.date = e.date

-- ft_max_date = as_of (обучение) либо as_of + horizon (инференс).
-- На инференсе последние 28 дней имеют sales = NULL — это и есть строки,
-- по которым модель будет предсказывать.
WHERE e.date <= DATE '{{ ft_max_date }}'
  AND {{ store_filter }}
  -- строки, где товара ещё не было в ассортименте, в обучение не идут
  AND e.days_since_first_sale >= 0;


-- Витрина прогнозов: сюда пишет inference, отсюда читают API и дашборд.
CREATE SCHEMA IF NOT EXISTS mart;

CREATE TABLE IF NOT EXISTS mart.forecast (
    id             VARCHAR   NOT NULL,
    item_id        VARCHAR   NOT NULL,
    store_id       VARCHAR   NOT NULL,
    date           DATE      NOT NULL,
    horizon        SMALLINT  NOT NULL,   -- 1..28
    prediction     FLOAT     NOT NULL,
    model_version  VARCHAR   NOT NULL,
    run_id         VARCHAR   NOT NULL,
    as_of          DATE      NOT NULL,   -- дата, на которую делался прогноз
    created_at     TIMESTAMP NOT NULL,
    PRIMARY KEY (id, date, as_of)        -- идемпотентность: перезапуск DAG не задваивает
);


-- Фактическое качество: заполняется, когда приезжают реальные продажи.
CREATE TABLE IF NOT EXISTS mart.forecast_accuracy (
    as_of          DATE      NOT NULL,
    date           DATE      NOT NULL,
    horizon        SMALLINT  NOT NULL,
    level          VARCHAR   NOT NULL,   -- total / state / store / dept / item
    level_value    VARCHAR   NOT NULL,
    mae            FLOAT,
    wmape          FLOAT,
    rmse           FLOAT,
    bias           FLOAT,
    model_version  VARCHAR,
    computed_at    TIMESTAMP NOT NULL,
    PRIMARY KEY (as_of, date, level, level_value)
);
