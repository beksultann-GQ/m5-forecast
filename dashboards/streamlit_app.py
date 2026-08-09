"""Дашборд мониторинга. Читает витрину, ничего не считает сам.

Аудитория — не дата-сайентист, а закупщик и дежурный. Поэтому первый экран
отвечает на два вопроса: «прогноз свежий?» и «ему можно верить?».
Графики важности фич и распределений — дальше, для тех, кому надо.

Запуск: make dashboard
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="M5 Forecast — мониторинг",
    page_icon="📦",
    layout="wide",
)


def page_overview() -> None:
    """Главный экран: свежесть прогноза и фактическое качество.

    Плитки сверху:
        - дата последнего прогноза и лаг в днях (красный, если > 1)
        - версия прод-модели и когда зарегистрирована
        - WMAPE за последние 28 дней + дельта к предыдущему периоду
        - bias: систематически недо- или перепрогнозируем

    TODO: читать mart.forecast и mart.forecast_accuracy.
    """
    raise NotImplementedError


def page_forecast_explorer() -> None:
    """Просмотр прогноза по паре (item, store).

    График: факт за прошлые 90 дней + прогноз на 28 вперёд.
    Фильтры: магазин, отдел, товар.

    TODO: реализовать.
    """
    raise NotImplementedError


def page_accuracy() -> None:
    """История фактического качества.

    Разрезы: по горизонту (1-7 vs 22-28), по категориям, по магазинам.
    Отдельно — топ худших рядов: закупщику интересны конкретные позиции,
    а не среднее по больнице.

    TODO: реализовать.
    """
    raise NotImplementedError


def page_drift() -> None:
    """Дрифт данных и прогнозов.

    Отдельным блоком — состав погодных фич по источнику
    (archive / forecast / climatology). Если доля climatology выросла,
    значит внешний API отвалился, и качество на дальнем горизонте просядет.

    TODO: реализовать, встроить html-отчёт Evidently через components.html.
    """
    raise NotImplementedError


def page_external_data() -> None:
    """Внешние данные: погода и праздники.

    Что показываем:
        - свежесть каждого источника
        - вклад внешних фич в модель (feature importance)
        - результат ablation: сколько дал каждый источник на бэктесте

    Последнее — главное. Если ablation показывает 0, внешние данные надо
    убирать, а не оставлять «потому что было интересно их подключить».

    TODO: реализовать.
    """
    raise NotImplementedError


PAGES = {
    "Обзор": page_overview,
    "Прогноз": page_forecast_explorer,
    "Качество": page_accuracy,
    "Дрифт": page_drift,
    "Внешние данные": page_external_data,
}


def main() -> None:
    st.sidebar.title("M5 Forecast")
    choice = st.sidebar.radio("Раздел", list(PAGES))
    st.sidebar.divider()
    st.sidebar.caption("Данные: DuckDB витрина mart.*")
    PAGES[choice]()


if __name__ == "__main__":
    main()
