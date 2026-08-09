"""Rolling origin CV. Самый важный кусок проекта.

Никакого random split. Никакого KFold. Только сдвиг по времени.

Схема одного фолда:

    |------------- train -------------|-- gap 28 --|--- val 28 ---|
                                    T                            T+56

    gap обязателен. Без него фичи вида roll_mean_7, посчитанные на границе,
    захватывают дни, которые попадут в val — модель увидит будущее.
    С gap=28 (равным горизонту) такой возможности нет по построению.

Фолды идут «лесенкой» назад от последней доступной даты:

    fold 0:  train до 2016-04-24 | val 2016-05-23..2016-06-19
    fold 1:  train до 2016-03-27 | val 2016-04-25..2016-05-22
    fold 2:  train до 2016-02-28 | val 2016-03-28..2016-04-24
    fold 3:  train до 2016-01-31 | val 2016-02-29..2016-03-27

Обучающее окно расширяющееся (expanding), а не скользящее: в ритейле
старая история несёт годовую сезонность, выкидывать её не надо.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from m5.config import Config
from m5.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    """Один фолд rolling-origin валидации."""

    index: int
    train_start: date
    train_end: date  # он же as_of: граница знания
    val_start: date  # train_end + gap + 1
    val_end: date  # val_start + horizon - 1

    def __post_init__(self) -> None:
        if self.train_end >= self.val_start:
            raise ValueError(
                f"Фолд {self.index}: train_end ({self.train_end}) "
                f">= val_start ({self.val_start}) — утечка по построению"
            )

    @property
    def gap_days(self) -> int:
        return (self.val_start - self.train_end).days - 1

    @property
    def horizon_days(self) -> int:
        return (self.val_end - self.val_start).days + 1

    @property
    def forecast_origin(self) -> date:
        """День, из которого делается прогноз на валидационное окно.

        Не путать с train_end. Это две разные вещи:
            train_end       — последняя строка, на которой УЧИМСЯ (отделена gap'ом)
            forecast_origin — момент, из которого ПРОГНОЗИРУЕМ (день перед val)

        В проде мы знаем продажи по вчера и предсказываем 28 дней вперёд, поэтому
        origin = val_start - 1. Утечки нет: все фичи используют лаги >= 28,
        а значит для даты val_start + k они смотрят не позже val_start + k - 28,
        что при k <= 27 не превышает forecast_origin.
        """
        return self.val_start - timedelta(days=1)

    def describe(self) -> str:
        return (
            f"fold={self.index} train=[{self.train_start}..{self.train_end}] "
            f"gap={self.gap_days}d val=[{self.val_start}..{self.val_end}]"
        )


class RollingOriginSplitter:
    """Генератор фолдов.

    Args:
        n_folds: сколько фолдов
        horizon: длина валидационного окна, дней
        gap_days: разрыв между train и val (>= horizon!)
        step_days: на сколько сдвигать origin между фолдами
        min_train_days: минимальная длина обучающего окна
        expanding: True — окно расширяется, False — скользит фиксированной длиной
    """

    def __init__(
        self,
        n_folds: int = 4,
        horizon: int = 28,
        gap_days: int = 28,
        step_days: int = 28,
        min_train_days: int = 730,
        expanding: bool = True,
    ) -> None:
        if gap_days < horizon:
            raise ValueError(
                f"gap_days ({gap_days}) < horizon ({horizon}): лаги протекут в валидацию"
            )
        self.n_folds = n_folds
        self.horizon = horizon
        self.gap_days = gap_days
        self.step_days = step_days
        self.min_train_days = min_train_days
        self.expanding = expanding

    def split(self, data_start: date, data_end: date) -> list[Fold]:
        """Построить фолды на отрезке [data_start, data_end].

        Фолд 0 заканчивается на data_end, остальные — «лесенкой» назад с шагом
        step_days. Так самый свежий отрезок всегда участвует в оценке.

        Raises:
            ValueError: если истории не хватает на n_folds фолдов.
        """
        folds: list[Fold] = []

        for index in range(self.n_folds):
            val_end = data_end - timedelta(days=index * self.step_days)
            val_start = val_end - timedelta(days=self.horizon - 1)
            train_end = val_start - timedelta(days=self.gap_days + 1)

            if self.expanding:
                train_start = data_start
            else:
                train_start = train_end - timedelta(days=self.min_train_days - 1)

            n_train_days = (train_end - train_start).days + 1
            if n_train_days < self.min_train_days:
                raise ValueError(
                    f"На фолд {index} остаётся {n_train_days} дней обучения "
                    f"при min_train_days={self.min_train_days}.\n"
                    f"Истории [{data_start}..{data_end}] хватает не на все {self.n_folds} фолдов. "
                    f"Варианты: уменьшить validation.n_folds, снизить min_train_days "
                    f"или взять более длинную историю."
                )

            fold = Fold(
                index=index,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
            )
            assert_fold_integrity(fold, self.horizon)
            folds.append(fold)
            logger.info(fold.describe())

        return folds


def make_folds(cfg: Config, data_start: date, data_end: date) -> list[Fold]:
    """Собрать фолды по конфигу."""
    splitter = RollingOriginSplitter(
        n_folds=cfg.validation.n_folds,
        horizon=cfg.validation.horizon,
        gap_days=cfg.validation.gap_days,
        step_days=cfg.validation.step_days,
        min_train_days=cfg.validation.min_train_days,
    )
    return splitter.split(data_start, data_end)


def assert_fold_integrity(fold: Fold, horizon: int) -> None:
    """Проверки целостности фолда. Вызывать перед КАЖДЫМ обучением.

    Дешёвая страховка от самой дорогой ошибки в проекте.

    Проверяем:
        1. train_end < val_start
        2. gap >= horizon
        3. длина val == horizon
        4. train не пустой
    """
    if fold.train_end >= fold.val_start:
        raise ValueError(f"{fold.describe()}: train пересекается с val")
    if fold.gap_days < horizon:
        raise ValueError(f"{fold.describe()}: gap {fold.gap_days} < horizon {horizon}")
    if fold.horizon_days != horizon:
        raise ValueError(f"{fold.describe()}: длина val {fold.horizon_days} != {horizon}")
    if fold.train_start >= fold.train_end:
        raise ValueError(f"{fold.describe()}: пустое обучающее окно")


def day_index_to_date(origin_date: date, day_index: int) -> date:
    """'d_1914' -> дата. d_1 соответствует origin_date."""
    return origin_date + timedelta(days=day_index - 1)


def date_to_day_index(origin_date: date, target: date) -> int:
    """Обратное преобразование."""
    return (target - origin_date).days + 1
