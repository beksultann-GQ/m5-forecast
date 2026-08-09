-- Staging: разворот продаж wide -> long.
--
-- Вход:  raw/sales_train_evaluation.parquet — 30490 строк × (6 ключей + 1941 колонка d_*)
-- Выход: stg.sales_long — ~59 млн строк (30490 × 1941)
--
-- Почему UNPIVOT, а не pandas.melt: melt материализует всё в RAM (десятки ГБ),
-- DuckDB стримит и спиллит на диск при нехватке памяти.

CREATE SCHEMA IF NOT EXISTS stg;

-- Иерархия товаров: отдельной маленькой таблицей, чтобы не таскать строки
-- item_id/dept_id/... через 59 млн строк.
CREATE OR REPLACE TABLE stg.hierarchy AS
SELECT DISTINCT
    id,
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id
FROM read_parquet('{{ raw_dir }}/sales_train_evaluation.parquet');


CREATE OR REPLACE TABLE stg.sales_long AS
WITH unpivoted AS (
    UNPIVOT (
        SELECT * FROM read_parquet('{{ raw_dir }}/sales_train_evaluation.parquet')
    )
    ON COLUMNS (* EXCLUDE (id, item_id, dept_id, cat_id, store_id, state_id))
    INTO NAME d VALUE sales
)
SELECT
    u.id,
    CAST(REPLACE(u.d, 'd_', '') AS INTEGER) AS day_index,
    c.date                                  AS date,
    CAST(u.sales AS SMALLINT)               AS sales
FROM unpivoted u
JOIN stg.calendar c USING (d);


-- Дата первой ненулевой продажи по ряду.
-- Нули ДО неё — товара не было в ассортименте, а не нулевой спрос.
-- Обучаться на них нельзя: модель выучит «этот товар не продаётся».
CREATE OR REPLACE TABLE stg.first_sale AS
SELECT
    id,
    MIN(date) FILTER (WHERE sales > 0) AS first_sale_date,
    MAX(date) FILTER (WHERE sales > 0) AS last_sale_date,
    COUNT(*) FILTER (WHERE sales > 0)  AS n_nonzero_days
FROM stg.sales_long
GROUP BY id;
