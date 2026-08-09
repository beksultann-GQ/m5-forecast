-- Staging: цены.
--
-- В raw цены недельные (ключ wm_yr_wk), продажи — дневные.
-- Здесь готовим недельный слой с производными признаками,
-- разворот в дневной делаем в ft-слое джойном по wm_yr_wk.
--
-- Важно: цена появляется в справочнике ТОЛЬКО когда товар есть в магазине.
-- Отсутствие цены = товара нет в продаже. Это сигнал, а не пропуск.

CREATE SCHEMA IF NOT EXISTS stg;

CREATE OR REPLACE TABLE stg.prices AS
WITH src AS (
    SELECT
        store_id,
        item_id,
        wm_yr_wk,
        CAST(sell_price AS FLOAT) AS sell_price
    FROM read_parquet('{{ raw_dir }}/sell_prices.parquet')
),
with_lags AS (
    SELECT
        *,
        LAG(sell_price, 1) OVER w  AS price_prev_week,
        LAG(sell_price, 4) OVER w  AS price_4w_ago,
        MIN(wm_yr_wk)     OVER (PARTITION BY store_id, item_id) AS first_price_week,
        AVG(sell_price)   OVER (PARTITION BY store_id, item_id) AS price_mean_item_store,
        STDDEV_POP(sell_price) OVER (PARTITION BY store_id, item_id) AS price_std_item_store,
        MAX(sell_price)   OVER (PARTITION BY store_id, item_id) AS price_max_item_store,
        COUNT(DISTINCT sell_price) OVER (PARTITION BY store_id, item_id) AS price_nunique_item
    FROM src
    WINDOW w AS (PARTITION BY store_id, item_id ORDER BY wm_yr_wk)
)
SELECT
    store_id,
    item_id,
    wm_yr_wk,
    sell_price,

    -- изменение цены к прошлой неделе: промо и подорожания двигают спрос
    sell_price / NULLIF(price_prev_week, 0) - 1        AS price_change_1w,
    sell_price / NULLIF(price_4w_ago, 0) - 1           AS price_momentum_4w,
    CASE WHEN price_prev_week IS NULL THEN 0
         WHEN sell_price < price_prev_week THEN 1
         ELSE 0 END                                    AS is_price_drop,

    -- позиция цены относительно собственной истории
    sell_price / NULLIF(price_mean_item_store, 0)      AS price_rel_own_mean,
    sell_price / NULLIF(price_max_item_store, 0)       AS price_rel_own_max,
    price_std_item_store,
    price_nunique_item,
    first_price_week
FROM with_lags;


-- Позиция цены товара относительно категории/отдела в ту же неделю:
-- «дорогой ли этот товар прямо сейчас на фоне соседей по полке».
CREATE OR REPLACE TABLE stg.prices_relative AS
SELECT
    p.store_id,
    p.item_id,
    p.wm_yr_wk,
    p.sell_price / NULLIF(AVG(p.sell_price) OVER (PARTITION BY p.wm_yr_wk, h.dept_id), 0)
        AS price_rel_dept_mean,
    p.sell_price / NULLIF(AVG(p.sell_price) OVER (PARTITION BY p.wm_yr_wk, h.cat_id), 0)
        AS price_rel_cat_mean,
    p.sell_price / NULLIF(AVG(p.sell_price) OVER (PARTITION BY p.wm_yr_wk, p.item_id), 0)
        AS price_rel_item_mean   -- этот же товар в других магазинах
FROM stg.prices p
JOIN (SELECT DISTINCT item_id, dept_id, cat_id FROM stg.hierarchy) h
    USING (item_id);
