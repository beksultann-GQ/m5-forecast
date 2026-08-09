-- Фичи: скользящие статистики.
--
-- ТО ЖЕ ПРАВИЛО: окно считается по ряду, СДВИНУТОМУ на горизонт.
-- Схема такая: сначала LAG(sales, 28), потом окно по этому сдвинутому ряду.
-- rolling_mean_7 по исходному ряду = утечка. rolling_mean_7 по lag_28 = честно.
--
-- В DuckDB это выражается как окно с рамкой ROWS BETWEEN 34 PRECEDING AND 28 PRECEDING
-- (7 дней, заканчивающихся 28 дней назад).

CREATE SCHEMA IF NOT EXISTS ft;

CREATE OR REPLACE TABLE ft.rolling AS
SELECT
    id,
    date,

    -- окно 7 дней, сдвинутое на 28
    AVG(sales)        OVER (w ROWS BETWEEN 34 PRECEDING  AND 28 PRECEDING) AS roll_mean_7,
    STDDEV_POP(sales) OVER (w ROWS BETWEEN 34 PRECEDING  AND 28 PRECEDING) AS roll_std_7,
    MAX(sales)        OVER (w ROWS BETWEEN 34 PRECEDING  AND 28 PRECEDING) AS roll_max_7,

    -- окно 28 дней, сдвинутое на 28
    AVG(sales)        OVER (w ROWS BETWEEN 55 PRECEDING  AND 28 PRECEDING) AS roll_mean_28,
    STDDEV_POP(sales) OVER (w ROWS BETWEEN 55 PRECEDING  AND 28 PRECEDING) AS roll_std_28,
    MAX(sales)        OVER (w ROWS BETWEEN 55 PRECEDING  AND 28 PRECEDING) AS roll_max_28,

    -- окно 180 дней, сдвинутое на 28 — «базовый уровень» ряда
    AVG(sales)        OVER (w ROWS BETWEEN 207 PRECEDING AND 28 PRECEDING) AS roll_mean_180,
    STDDEV_POP(sales) OVER (w ROWS BETWEEN 207 PRECEDING AND 28 PRECEDING) AS roll_std_180,

    -- доля нулей: прерывистость спроса. Для M5 критично — ~68% строк нулевые,
    -- и модели полезно знать, редкий это товар или ходовой.
    AVG(CASE WHEN sales = 0 THEN 1.0 ELSE 0.0 END)
        OVER (w ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING) AS zero_ratio_28,
    AVG(CASE WHEN sales = 0 THEN 1.0 ELSE 0.0 END)
        OVER (w ROWS BETWEEN 207 PRECEDING AND 28 PRECEDING) AS zero_ratio_180,

    -- сколько дней подряд не продавалось на момент отсечки
    COUNT(*) FILTER (WHERE sales > 0)
        OVER (w ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING) AS nonzero_days_28,

    -- сезонность дня недели: средние продажи в этот же день недели,
    -- окно из ~8 недель, сдвинутое на горизонт
    AVG(sales) OVER (
        PARTITION BY id, dayofweek ORDER BY date
        ROWS BETWEEN 11 PRECEDING AND 4 PRECEDING
    ) AS dow_mean_8w,

    -- тренд: отношение недавнего уровня к базовому
    AVG(sales) OVER (w ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING)
        / NULLIF(AVG(sales) OVER (w ROWS BETWEEN 207 PRECEDING AND 28 PRECEDING), 0)
        AS trend_28_over_180

FROM stg.sales_enriched
WHERE date <= DATE '{{ as_of }}'
WINDOW w AS (PARTITION BY id ORDER BY date);


-- Агрегаты уровнем выше: продажи отдела/категории в магазине.
-- Ряд одного товара шумный, а уровень отдела стабильный — модель ловит
-- общий спад/подъём трафика в магазине. Сдвиг на горизонт обязателен и здесь.
CREATE OR REPLACE TABLE ft.rolling_agg AS
WITH dept_daily AS (
    SELECT store_id, dept_id, date, SUM(sales) AS dept_sales
    FROM stg.sales_enriched
    WHERE date <= DATE '{{ as_of }}'
    GROUP BY 1, 2, 3
),
cat_daily AS (
    SELECT store_id, cat_id, date, SUM(sales) AS cat_sales
    FROM stg.sales_enriched
    WHERE date <= DATE '{{ as_of }}'
    GROUP BY 1, 2, 3
)
SELECT
    d.store_id,
    d.dept_id,
    c.cat_id,
    d.date,
    AVG(d.dept_sales) OVER (
        PARTITION BY d.store_id, d.dept_id ORDER BY d.date
        ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING
    ) AS dept_roll_mean_28,
    AVG(c.cat_sales) OVER (
        PARTITION BY c.store_id, c.cat_id ORDER BY c.date
        ROWS BETWEEN 55 PRECEDING AND 28 PRECEDING
    ) AS cat_roll_mean_28
FROM dept_daily d
JOIN (SELECT DISTINCT dept_id, cat_id FROM stg.hierarchy) m USING (dept_id)
JOIN cat_daily c ON c.store_id = d.store_id AND c.cat_id = m.cat_id AND c.date = d.date;
