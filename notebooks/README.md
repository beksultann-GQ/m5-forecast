**English** · [Қазақша](README.kk.md) · [Русский](README.ru.md)

# Notebooks

## `00_defence.ipynb` — project presentation (5–7 minutes)

The only notebook committed **together with its output**: the plots and tables
in it are exactly what gets shown. It reads the data mart in read-only mode and
changes nothing.

Run: `uv run jupyter lab notebooks/00_defence.ipynb`, or open it in VS Code
and pick the kernel from `.venv`.

Before presenting, do a `Run All` — the numbers will be pulled from the current
state of the mart instead of staying hard-coded.

---

## Everything else — EDA and drafts only

Nothing written here is part of the pipeline.

**Rules:**

- Logic that turned out useful moves to `src/m5/` and gets covered by a test.
  A notebook is a draft, not the place where code lives.
- Cell output is stripped before committing (`nbstripout` in pre-commit does it).
- Data from notebooks is not saved to `data/processed` — only via the CLI.

**What is worth looking at in EDA:**

1. `01_eda_sales.ipynb` — sales distribution, share of zeros by category,
   what "intermittent" series look like, when items appear in the assortment.
2. `02_eda_calendar_snap.ipynb` — the effect of SNAP days (benefit payment days)
   by state and category. Hypothesis: strongest for FOODS.
3. `03_eda_prices.ipynb` — how often prices change, whether there is a visible
   sales response to price cuts.
4. `04_eda_external.ipynb` — checking hypotheses on external data:
   correlation of sales with temperature by category, sales around holidays,
   cross-check of Nager.Date holidays against `event_name_1/2` from M5.

Item 4 is the most important: if there is no correlation even visually, it is
not worth spending time wiring an external source into the pipeline.
