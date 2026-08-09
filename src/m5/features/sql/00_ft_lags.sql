-- Фичи: лаги продаж.
--
-- ПРАВИЛО, НАРУШЕНИЕ КОТОРОГО УБИВАЕТ ПРОЕКТ:
--   минимальный лаг >= горизонт прогноза (28).
--   Если взять lag_7, то для предсказания дня T+28 понадобятся продажи дня T+21,
--   которых в момент прогноза ещё не существует. На валидации это даёт
--   прекрасный скор, в проде — катастрофу.
--
-- Поэтому лаги: 28, 35, 42, 56, 364. Никаких lag_1 / lag_7.

CREATE SCHEMA IF NOT EXISTS ft;

CREATE OR REPLACE TABLE ft.lags AS
SELECT
    id,
    date,

    LAG(sales, 28)  OVER w AS sales_lag_28,
    LAG(sales, 35)  OVER w AS sales_lag_35,
    LAG(sales, 42)  OVER w AS sales_lag_42,
    LAG(sales, 56)  OVER w AS sales_lag_56,
    LAG(sales, 364) OVER w AS sales_lag_364,   -- тот же день недели год назад

    -- Динамика: ускоряется ряд или затухает
    LAG(sales, 28) OVER w - LAG(sales, 35) OVER w  AS sales_diff_28_35,
    LAG(sales, 28) OVER w - LAG(sales, 56) OVER w  AS sales_diff_28_56,

    -- Тот же день недели неделями раньше (лаги кратные 7 сохраняют недельную сезонность)
    (LAG(sales, 28) OVER w + LAG(sales, 35) OVER w + LAG(sales, 42) OVER w) / 3.0
        AS sales_same_dow_mean_3w

FROM stg.sales_enriched
WHERE date <= DATE '{{ as_of }}'
WINDOW w AS (PARTITION BY id ORDER BY date);
