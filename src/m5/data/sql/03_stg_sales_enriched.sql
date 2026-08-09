-- Staging: продажи + иерархия + календарь + цены в одной таблице.
-- Это вход для ft-слоя. Здесь ещё НЕТ лагов и окон — только «сырые» атрибуты дня.
--
-- Две обрезки:
--   1. keep_last_days   — не тащим всю историю с 2011, если она не нужна
--   2. first_sale_date  — выкидываем нули до появления товара в ассортименте

CREATE SCHEMA IF NOT EXISTS stg;

CREATE OR REPLACE TABLE stg.sales_enriched AS
SELECT
    s.id,
    h.item_id,
    h.dept_id,
    h.cat_id,
    h.store_id,
    h.state_id,
    s.date,
    s.day_index,
    s.sales,

    -- календарь
    c.wm_yr_wk,
    c.dayofweek,
    c.dayofmonth,
    c.weekofyear,
    c.dayofyear,
    c.month,
    c.year,
    c.quarter,
    c.is_weekend,
    c.is_month_start,
    c.is_month_end,
    c.event_name_1,
    c.event_type_1,
    c.event_name_2,
    c.event_type_2,
    c.has_event,
    ce.days_to_next_event,
    ce.days_since_prev_event,

    -- SNAP по штату магазина (дни выплат пособий — сильно двигают продажи еды)
    sn.snap,

    -- цены
    p.sell_price,
    p.price_change_1w,
    p.price_momentum_4w,
    p.is_price_drop,
    p.price_rel_own_mean,
    p.price_rel_own_max,
    p.price_std_item_store,
    p.price_nunique_item,
    pr.price_rel_item_mean,
    pr.price_rel_dept_mean,
    pr.price_rel_cat_mean,

    -- «товар уже в ассортименте»
    fs.first_sale_date,
    DATEDIFF('day', fs.first_sale_date, s.date) AS days_since_first_sale,
    CASE WHEN p.sell_price IS NULL THEN 1 ELSE 0 END AS is_out_of_assortment

FROM stg.sales_long s
JOIN stg.hierarchy h        USING (id)
JOIN stg.calendar  c        USING (date)
LEFT JOIN stg.calendar_events ce USING (date)
LEFT JOIN stg.snap sn       ON sn.date = s.date AND sn.state_id = h.state_id
LEFT JOIN stg.prices p      ON p.store_id = h.store_id
                           AND p.item_id  = h.item_id
                           AND p.wm_yr_wk = c.wm_yr_wk
LEFT JOIN stg.prices_relative pr ON pr.store_id = h.store_id
                                AND pr.item_id  = h.item_id
                                AND pr.wm_yr_wk = c.wm_yr_wk
LEFT JOIN stg.first_sale fs USING (id)

WHERE
    -- граница считается из data.keep_last_days; null в конфиге = вся история
    s.date >= DATE '{{ history_start_date }}'
    -- нули до первой продажи в обучение не идут
    AND (fs.first_sale_date IS NULL OR s.date >= fs.first_sale_date);
