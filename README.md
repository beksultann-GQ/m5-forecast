**English** · [Қазақша](README.kk.md) · [Русский](README.ru.md)

# M5 Forecast — daily sales forecasting

A production-ready demand forecasting pipeline on the [M5 Forecasting Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) data.

> **Status: a working thin vertical slice.**
> The data layer, feature mart, baselines and LightGBM training are implemented and verified.
> `make demo` builds everything on synthetic data in ~20 seconds, no Kaggle account needed.
> Inference into the mart, WRMSSE, MLflow Registry, Airflow tasks and external data
> are still skeletons with `TODO`s. What comes next is in the [Roadmap](#roadmap).

---

## Problem statement

This is not a competition entry but a product task. The formulation was fixed **before** any code was written.

**What we forecast.** Daily unit sales per (item, store) pair. Horizon — 28 days.
30,490 series: 3,049 items × 10 stores in 3 US states (CA, TX, WI).

**Who the consumer is.** A hypothetical purchasing department. It needs a batch forecast once a week
to fit the supplier ordering cycle, not a realtime API. Every architectural decision follows from that:
batch inference into a mart, FastAPI serves precomputed results, the model computes nothing on the fly.

**Metrics.**

| Metric | Role |
|---|---|
| **WRMSSE** | The competition metric. Weighted across 12 aggregation levels. Needed to compare against the public leaderboard and understand where we stand globally. |
| **WMAPE** | The business metric. "We are off by 23% of volume" — a buyer understands that. MAPE is not applicable: ~68% of values are zero. |
| **Bias** | Systematic under-/over-forecasting. Under-forecast = lost sales, over-forecast = frozen cash and write-offs. The sign matters more than the magnitude. |

**The baseline to beat.** `seasonal_naive`: sales = exactly 28 days ago.
Plus `dow_mean` (mean by day of week over 8 weeks) as a more honest bar.

Without a baseline any ML is cargo cult. The number obtained here is the bar for the whole project.

---

## Architectural decisions and why

**Feature engineering in SQL (DuckDB), Python is the glue.**
After unpivoting to long format this is ~59M rows. `pd.melt` on such a dataset simply does not work
on a laptop. DuckDB does `UNPIVOT` + window functions out-of-core.
It is also closer to how features are computed in a real DWH.

**One function computes features for both train and inference.**
Not two scripts. The only difference is the `as_of` parameter. Otherwise you get
training/serving skew — the classic production bug where validation looks great
and production is garbage because someone computed the rolling mean slightly differently in the inference script.

**All lags and windows are shifted by at least 28 days.**
The forecast horizon is 28 days. So at forecast time the last 28 days of sales
are not yet known. `lag_7` is physically impossible to compute in production, yet on validation
it gives a superb score. This is mistake #1 in time series forecasting, and it is silent.

**Direct, not recursive.** Recursive (predict day 1, plug it into the features,
predict day 2) by day 28 builds the forecast almost entirely on its own
predictions and accumulates error. Here every day of the horizon is predicted
directly from features known at forecast time.

An important caveat worth saying out loud: **horizon groups
(1-7, 8-14, ...) are meaningless with the current feature set.** Since all lags are >= 28
and computed relative to the row date, one feature vector is valid for the whole horizon —
four models would train on identical data. Direct multi-step pays off
when near horizons get fresher features: for h=1..7 `lag_7` is legitimate.
That requires parameterising the minimum lag in SQL and building four marts —
the next step, not something already done.

**LightGBM with objective=tweedie.**
~68% of the target is zeros. Tweedie is a mixture of Poisson (how many purchases) and gamma (purchase size),
exactly the structure of intermittent retail demand. On M5 it consistently beats RMSE.

**10 models, one per store.**
Lower peak memory (fits on a MacBook), plus the model captures store specifics.
In total an ensemble of 10 × 4 = 40 small models.

**No neural networks.** Until the whole pipeline works, N-BEATS and the like
add no value to the project and would eat a month.

---

## Additional data sources

On top of the Kaggle dataset two external APIs are wired in. Both free, no key required.

### Weather — Open-Meteo (fallback: Visual Crossing)

**Hypothesis.** Weather moves sales: heat → drinks and ice cream, snowfall → a dip
in store traffic, a sharp cold snap → stocking up. For FOODS (the largest
M5 category) the effect should be noticeable.

**What we take.** Daily aggregates for one representative point per state
(LA / Dallas / Madison). Walmart does not publish exact store addresses —
this is a deliberate simplification, and it should be stated, not hidden.

**Derived features.** Raw temperature is a weak signal. What works is **deviation from the norm**:
+25 °C in January in Wisconsin and +25 °C in July in California mean completely different things.
We compute `temp_anomaly`, degree-days `hdd`/`cdd`, flags `is_heavy_rain` / `is_snow_day`,
the swing `temp_change_1d`.

**The main trap — training/serving skew.**
In training the actual weather is known. At inference, over a 28-day horizon,
it is not known and cannot be. Strategy:

```
days 1–16  →  Open-Meteo Forecast API
days 17–28 →  climatological norm by day of year
```

The `weather_source` column (`archive` / `forecast` / `climatology`) travels all the way to the mart
and is monitored separately. If the share of `climatology` jumps — the external API is down,
and quality on the far horizon will degrade before the actuals arrive.

The honest option is to train directly on "forecast" weather, but we have no historical forecasts.
This is a recorded limitation, not a forgotten detail.

### Public holidays — Nager.Date

**Why, when `calendar.csv` already has `event_name_1/2`:**

1. M5 events are a flat list with no regionality. Cesar Chavez Day is a holiday
   in California and an ordinary working day in Texas. Nager returns holidays
   by subdivision (`US-CA`, `US-TX`, `US-WI`).
2. Nager distinguishes types: Public / Bank / School / Optional — the effect on traffic differs.
3. An independent source: it can be cross-checked to find gaps in M5 events
   (the `m5 external compare-events` command).
4. **No skew.** Unlike weather, the holiday calendar is known years ahead
   and equally available in train and in production. It is a "free" feature.

**Derived features.** The holiday itself is a weak signal: the store is half-empty or closed.
What works is the neighbourhood — people stock up **the day before**. We compute `is_day_before_holiday`,
`is_day_after_holiday`, `days_to_next_holiday`, `is_long_weekend`.

### Checking that it actually helped

External data is not added "for solidity". The project includes an ablation:

```bash
make backtest                       # base
m5 model backtest --ablation        # base / +weather / +holidays / all
```

The comparison runs on the same folds. If the gain is zero — the source is removed.
The result of that table should live in the README: it is the answer to the question
"why did you take external data at all".

---

## Validation

No random split. Only **rolling origin** in time.

```
|-------------- train --------------|--- gap 28 ---|--- val 28 ---|
                                   T                              T+56
```

**The 28-day gap is mandatory.** Without it, rolling statistics computed
at the train boundary capture days from val. With `gap = horizon` that is
impossible by construction.

4 folds stepping back from the last date, an expanding training window
(old history carries yearly seasonality — no need to throw it away).

Tests cover this: [tests/test_leakage.py](tests/test_leakage.py). The key one is
`test_feature_value_stable_when_future_changes`: we change sales after `as_of`
and check that features before `as_of` are byte-for-byte unchanged. If they changed —
some feature looks into the future, and it does not matter which.

---

## Stack

| Layer | Technology |
|---|---|
| Data loading | Kaggle CLI → Parquet |
| Storage / transformations | DuckDB (layers `raw` → `stg` → `ft` → `mart`) |
| Data validation | pandera + SQL invariants |
| Features | SQL (DuckDB window functions) |
| External data | Open-Meteo, Nager.Date (httpx + file cache) |
| Model | LightGBM, objective=tweedie |
| Experiments | MLflow + Model Registry |
| Orchestration | Airflow (3 DAGs) |
| Monitoring | Evidently + Streamlit |
| CI/CD | pytest, ruff, GitHub Actions |
| Packaging | Docker, Docker Compose |
| API | FastAPI (optional) |

---

## Layout

```
.
├── conf/                  # yaml configs, nothing hard-coded in code
│   ├── config.yaml        #   root + defaults
│   ├── data.yaml          #   files, layers, cut-offs
│   ├── features.yaml      #   lags, windows, calendar, prices
│   ├── model.yaml         #   LightGBM, horizon groups
│   ├── validation.yaml    #   rolling origin, metrics
│   └── external.yaml      #   weather and holidays
├── src/m5/
│   ├── cli.py             # single entry point: both humans and Airflow call it
│   ├── config/            # config loading
│   ├── data/              # download, csv→parquet, staging, schemas
│   │   └── sql/           #   staging layer: UNPIVOT, calendar, prices
│   ├── external/          # Open-Meteo, Nager.Date, caching HTTP client
│   ├── features/          # feature mart build
│   │   └── sql/           #   lags, windows, weather, holidays, mart
│   ├── models/            # baseline, train, predict, MLflow registry
│   ├── evaluation/        # WRMSSE, rolling origin CV, backtest
│   ├── monitoring/        # drift, actual quality
│   └── serving/           # FastAPI on top of the mart
├── dags/                  # m5_training, m5_inference, m5_external_ingest
├── tests/
├── dashboards/            # Streamlit
├── docker/
└── notebooks/             # EDA and drafts only
```

---

## Quick start

```bash
make setup            # uv + venv + dependencies + pre-commit
brew install libomp   # macOS: OpenMP runtime, LightGBM does not import without it
```

**Without a Kaggle account** — a synthetic dataset of the same shape:

```bash
make demo             # synth -> parquet -> staging -> feature mart -> table list
make baseline         # the bar to beat
make train            # train LightGBM + show feature importance
make tables           # what is in DuckDB
make dbpath           # path to the file for DBeaver
```

**On real data** — the same commands, only download instead of `synth`:

```bash
cp .env.example .env  # fill in KAGGLE_USERNAME / KAGGLE_KEY
make download         # ~450 MB from Kaggle
make raw staging validate features baseline train
```

Not implemented yet, will fail with `NotImplementedError`:
`make external`, `make backtest`, `make predict`, `make api`, `make dashboard`.

Services:

```bash
make up               # airflow (:8080) + mlflow (:5555) + postgres
make api              # FastAPI on :8000
make dashboard        # Streamlit on :8501
```

Development:

```bash
make lint
make fmt
make test             # without network and slow tests
make test-all
```

---

## Browsing data in DBeaver

Path to the file: `make dbpath`. In DBeaver — **New Database Connection → DuckDB**,
paste the path into **Path**.

**Mandatory**: on the *Driver properties* tab set `duckdb.read_only = true`.

Reason: DuckDB is an embedded database with a file lock. At any time there can be
**either one writer process, or any number of readers**. DBeaver connected
in write mode holds the file — and the pipeline crashes with
`Could not set lock on file`.

| Command | Mode | Coexists with DBeaver (read-only) |
|---|---|---|
| `make baseline`, `make train` | read | yes |
| `m5 db tables / peek / sql` | read | yes |
| `make staging`, `make features`, `make demo` | write | no, Disconnect first |

Browsing the mart during training is fine. Rebuilding it with DBeaver
connected is not.

---

## Airflow

**`m5_training`** — weekly, Sunday 02:00

```
ingest → validate → external → build_features → train → backtest
       → compare with the production model → register, IF better by 1%+
```

The last step is a gate. A new model does not go to production just because it is new.
A gain under 1% is noise between folds, not progress.

**`m5_inference`** — daily, 05:00

```
model from registry → features → 28-day forecast
       → sanity checks → mart → actual quality + drift
```

Sanity checks **before** writing to the mart: no NaN or negatives, row count equals
`n_series × 28`, volume within ±40% of the actuals for the previous 28 days.
A bad forecast is worse than no forecast — goods will be ordered based on it.

**`m5_external_ingest`** — daily, 03:00

A separate DAG: external APIs fail for their own reasons, and their failure must not
bring down training or inference — those will run on yesterday's cache.

All tasks are idempotent, with retries and alerts on failure.

---

## Roadmap

Do not go through the steps linearly to perfection. First — **a thin vertical slice in 1–2 weeks**:

```
download data → 5 features → LightGBM → one metric → one DAG → everything in Docker
```

Even with poor quality. Then deepen each layer.

Otherwise there is a high risk of getting stuck on feature engineering for two months
and never reaching the most valuable part — the pipeline.

Rough total: **6–10 weeks at 8–10 hours a week.**

**Done:**

1. ✅ `data/` + staging SQL → raw → stg in DuckDB, wide→long unpivot
2. ✅ `features/build.py` + ft SQL → feature mart, leakage check after every build
3. ✅ `evaluation/cv.py` → rolling origin with gap, tests on fold boundaries
4. ✅ `models/baseline.py` → the bar is fixed on 4 folds
5. ✅ `models/train.py` → LightGBM tweedie per store, beats the baseline

**Next, by priority:**

6. `evaluation/metrics.py::wrmsse` → the competition metric (currently only WMAPE/MAE)
7. `models/predict.py` → inference into `mart.forecast` + sanity checks
8. `evaluation/backtest.py` → run across all folds, not just fold zero
9. `models/registry.py` → MLflow Registry and the promotion gate
10. `external/` → weather and holidays + ablation, prove the gain
11. DAGs → bodies of `_beats_production` and `_sanity_checks`
12. `monitoring/`, `serving/`, dashboard

---

## Where it usually breaks

1. **Leakage through lags.** Computed a rolling mean without shifting by the horizon.
   Validation score is superb, test is a catastrophe. Cured by the tests
   in `test_leakage.py` and the rule "minimum lag >= horizon".
2. **Memory.** `pd.melt` on the whole dataset does not work. DuckDB or Polars.
3. **WRMSSE.** A non-trivial metric: 12 aggregation levels, weights by revenue
   over the last 28 days of train, the RMSSE denominator is computed only on train
   and only from the first non-zero sale. Understand it, do not copy it — otherwise
   you cannot explain it in an interview.
4. **Items not yet on sale.** Zeros before the first sale mean the item
   was not in the assortment, not zero demand. They must be cut off,
   otherwise the model learns from garbage.
5. **Weather at inference.** See the external data section: actual weather
   28 days ahead does not exist. If this is ignored, the backtest will show
   an improvement that will not happen in production.

---

## License and data

The M5 data belongs to the competition organisers and is not committed to the repository
(`data/` is in `.gitignore`). Open-Meteo — CC BY 4.0. Nager.Date — an open API.
