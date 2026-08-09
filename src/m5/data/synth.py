"""Генератор синтетического M5-датасета той же формы, что настоящий.

Зачем это в проекте, а не только в тестах:

    1. Пайплайн можно прогнать целиком, не дожидаясь скачивания 450 МБ с Kaggle
       и не имея Kaggle-аккаунта. Демо, онбординг нового человека, CI.
    2. Структура файлов идентична настоящей (те же имена колонок, тот же широкий
       формат d_1..d_N), поэтому весь код дальше по пайплайну не отличает
       синтетику от реальных данных.
    3. В данные заложены ИЗВЕСТНЫЕ эффекты (недельная сезонность, SNAP, реакция
       на цену, праздничные всплески) — на них видно, ловит их модель или нет.

Размер по умолчанию: 100 товаров × 5 магазинов × 800 дней = 400 тыс. строк
после разворота. Настоящий M5 — 59 млн. Форма та же, объём другой.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from m5.config import Config
from m5.utils.logging import get_logger
from m5.utils.paths import ensure_dir

logger = get_logger(__name__)

# Иерархия как в настоящем M5, только уже
CATEGORIES = {
    "FOODS": ["FOODS_1", "FOODS_2", "FOODS_3"],
    "HOUSEHOLD": ["HOUSEHOLD_1", "HOUSEHOLD_2"],
    "HOBBIES": ["HOBBIES_1"],
}

STORES = {
    "CA": ["CA_1", "CA_2"],
    "TX": ["TX_1", "TX_2"],
    "WI": ["WI_1"],
}

# События из настоящего calendar.csv — берём подмножество с реальными названиями
EVENTS = {
    (1, 1): ("NewYear", "National"),
    (2, 14): ("ValentinesDay", "Cultural"),
    (5, 31): ("MemorialDay", "National"),
    (7, 4): ("IndependenceDay", "National"),
    (9, 7): ("LaborDay", "National"),
    (11, 26): ("Thanksgiving", "National"),
    (12, 25): ("Christmas", "National"),
}


def generate(
    cfg: Config,
    n_items_per_dept: int = 20,
    n_days: int = 800,
    start_date: date = date(2014, 1, 1),
    seed: int = 42,
) -> dict[str, Path]:
    """Сгенерировать CSV-файлы в data/raw.

    Имена файлов и колонок совпадают с настоящим M5 — дальше по пайплайну
    разницы нет.

    Returns:
        {логическое_имя: путь_к_csv}
    """
    rng = np.random.default_rng(seed)
    raw_dir = ensure_dir(cfg.paths.raw_dir)

    calendar = _make_calendar(start_date, n_days)
    items = _make_items(n_items_per_dept)
    prices = _make_prices(items, calendar, rng)
    sales = _make_sales(items, calendar, prices, rng)

    paths = {
        "calendar": raw_dir / "calendar.csv",
        "sell_prices": raw_dir / "sell_prices.csv",
        "sales_train_evaluation": raw_dir / "sales_train_evaluation.csv",
        "sample_submission": raw_dir / "sample_submission.csv",
    }

    calendar.to_csv(paths["calendar"], index=False)
    prices.to_csv(paths["sell_prices"], index=False)
    sales.to_csv(paths["sales_train_evaluation"], index=False)
    _make_submission(sales).to_csv(paths["sample_submission"], index=False)

    logger.info(
        "Синтетика готова: %d рядов × %d дней = %s строк после разворота",
        len(sales),
        n_days,
        f"{len(sales) * n_days:,}",
    )
    return paths


def _make_calendar(start: date, n_days: int) -> pd.DataFrame:
    """Календарь в формате calendar.csv: d_1..d_N, события, SNAP-флаги."""
    dates = [start + timedelta(days=i) for i in range(n_days)]
    rows = []
    for i, d in enumerate(dates):
        event = EVENTS.get((d.month, d.day), ("", ""))
        rows.append(
            {
                "date": d.isoformat(),
                # неделя Walmart: YYWW, как в оригинале (11101 = 2011, неделя 01)
                "wm_yr_wk": int(f"1{d.year % 100:02d}{min(d.isocalendar()[1], 52):02d}"),
                "weekday": d.strftime("%A"),
                # в M5 wday: 1=Saturday .. 7=Friday
                "wday": (d.weekday() + 2) % 7 + 1,
                "month": d.month,
                "year": d.year,
                "d": f"d_{i + 1}",
                "event_name_1": event[0],
                "event_type_1": event[1],
                "event_name_2": "",
                "event_type_2": "",
                # SNAP: дни выплат пособий, в каждом штате свой график
                "snap_CA": int(d.day <= 10),
                "snap_TX": int(d.day in range(1, 16, 2)),
                "snap_WI": int(d.day in range(2, 16, 2)),
            }
        )
    return pd.DataFrame(rows)


def _make_items(n_per_dept: int) -> pd.DataFrame:
    """Иерархия товаров: item_id, dept_id, cat_id × магазины."""
    rows = []
    for cat_id, depts in CATEGORIES.items():
        for dept_id in depts:
            for i in range(n_per_dept):
                item_id = f"{dept_id}_{i + 1:03d}"
                for state_id, stores in STORES.items():
                    for store_id in stores:
                        rows.append(
                            {
                                "id": f"{item_id}_{store_id}_evaluation",
                                "item_id": item_id,
                                "dept_id": dept_id,
                                "cat_id": cat_id,
                                "store_id": store_id,
                                "state_id": state_id,
                            }
                        )
    return pd.DataFrame(rows)


def _make_prices(
    items: pd.DataFrame, calendar: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Недельные цены. У части товаров есть промо-провалы цены."""
    weeks = calendar["wm_yr_wk"].unique()
    pairs = items[["store_id", "item_id"]].drop_duplicates()

    rows = []
    for _, pair in pairs.iterrows():
        base = rng.uniform(0.5, 25.0)
        # товар появляется в ассортименте не сразу: цены нет до first_week
        first_week_idx = int(rng.choice([0, 0, 0, 0, len(weeks) // 4, len(weeks) // 2]))
        promo_weeks = set(rng.choice(len(weeks), size=max(1, len(weeks) // 20), replace=False))

        for week_idx, week in enumerate(weeks):
            if week_idx < first_week_idx:
                continue
            drift = 1.0 + week_idx / len(weeks) * rng.uniform(-0.1, 0.25)
            promo = 0.75 if week_idx in promo_weeks else 1.0
            rows.append(
                {
                    "store_id": pair.store_id,
                    "item_id": pair.item_id,
                    "wm_yr_wk": week,
                    "sell_price": round(base * drift * promo, 2),
                }
            )
    return pd.DataFrame(rows)


def _make_sales(
    items: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Продажи в ШИРОКОМ формате — как sales_train_evaluation.csv.

    Заложенные эффекты (модель должна их найти):
        - недельная сезонность: выходные выше буднего дня
        - годовая сезонность
        - SNAP-дни поднимают FOODS
        - снижение цены поднимает продажи
        - праздники: Thanksgiving/Christmas — всплеск, в сам день Рождества провал
        - прерывистость: много нулей, у части товаров продажи начинаются поздно
    """
    n_days = len(calendar)
    day_cols = calendar["d"].tolist()

    # dict, а не pandas-индекс: полмиллиона .get() по MultiIndex — это минуты
    price_lookup = {
        (row.store_id, row.item_id, row.wm_yr_wk): row.sell_price for row in prices.itertuples()
    }
    item_mean_price = prices.groupby(["store_id", "item_id"])["sell_price"].mean().to_dict()

    cal = calendar.to_dict("records")
    matrix = np.zeros((len(items), n_days), dtype=np.int16)

    for row_i, item in enumerate(items.itertuples()):
        base = rng.gamma(shape=1.5, scale=1.2)
        snap_col = f"snap_{item.state_id}"
        # чувствительность к цене: эластичность спроса
        elasticity = rng.uniform(-2.5, -0.5)
        mean_price = item_mean_price.get((item.store_id, item.item_id), float("nan"))

        for day_i, day in enumerate(cal):
            price = price_lookup.get((item.store_id, item.item_id, day["wm_yr_wk"]))
            if price is None:
                continue  # товара нет в ассортименте -> ноль, и это не спрос

            lam = base
            lam *= 1.35 if day["wday"] in (1, 2) else 1.0  # выходные
            lam *= 1.0 + 0.15 * np.sin(2 * np.pi * day_i / 365.25)  # годовая волна
            if item.cat_id == "FOODS" and day[snap_col]:
                lam *= 1.25  # SNAP двигает еду
            if not pd.isna(mean_price) and mean_price > 0:
                lam *= (price / mean_price) ** elasticity  # реакция на цену

            name = day["event_name_1"]
            if name == "Thanksgiving":
                lam *= 2.2
            elif name == "Christmas":
                lam *= 0.1  # магазин закрыт
            elif name:
                lam *= 1.3

            matrix[row_i, day_i] = rng.poisson(max(lam, 0.01))

    wide = pd.DataFrame(matrix, columns=day_cols)
    return pd.concat([items.reset_index(drop=True), wide], axis=1)


def _make_submission(sales: pd.DataFrame) -> pd.DataFrame:
    """sample_submission.csv: F1..F28 по validation- и evaluation-строкам."""
    ids = sales["id"].tolist()
    val_ids = [i.replace("_evaluation", "_validation") for i in ids]
    frame = pd.DataFrame({"id": val_ids + ids})
    for h in range(1, 29):
        frame[f"F{h}"] = 0
    return frame
