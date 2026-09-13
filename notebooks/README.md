# Notebooks

English · Қазақша · Русский — click a section to expand it.

<details open>
<summary><b>English</b></summary>

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

</details>

<details>
<summary><b>Қазақша</b></summary>

## `00_defence.ipynb` — жобаның презентациясы (5–7 минут)

**Нәтижесімен бірге** коммиттелетін жалғыз ноутбук: ондағы графиктер мен
кестелер — көрсетілетін нәрсенің өзі. Витринаны тек оқу (read-only) режимінде
оқиды, ештеңені өзгертпейді.

Іске қосу: `uv run jupyter lab notebooks/00_defence.ipynb` немесе VS Code-та
ашып, `.venv` ішінен ядроны таңдау.

Көрсетер алдында `Run All` жасаған жөн — сандар витринаның ағымдағы күйінен
тартылады, қатып қалған күйінде қалмайды.

---

## Қалғаны — тек EDA және қаралама

Мұнда жазылғанның ешқайсысы пайплайнның бөлігі емес.

**Ережелер:**

- Пайдалы болып шыққан логика `src/m5/` ішіне көшеді және тестпен жабылады.
  Ноутбук — қаралама, код тұратын орын емес.
- Ұяшықтардың нәтижесі коммит алдында тазаланады (pre-commit ішіндегі
  `nbstripout` осыны істейді).
- Ноутбуктегі деректер `data/processed` ішіне сақталмайды — тек CLI арқылы.

**EDA-да неге қарау керек:**

1. `01_eda_sales.ipynb` — сатылымның таралуы, категория бойынша нөлдердің
   үлесі, «үзік-үзік» қатарлар қалай көрінеді, тауарлар ассортиментке қашан
   пайда болады.
2. `02_eda_calendar_snap.ipynb` — SNAP күндерінің (жәрдемақы төленетін күндер)
   штаттар мен категориялар бойынша әсері. Гипотеза: ең күштісі FOODS-та.
3. `03_eda_prices.ipynb` — бағалар қаншалықты жиі өзгереді, бағаның төмендеуіне
   сатылымның көрінетін реакциясы бар ма.
4. `04_eda_external.ipynb` — сыртқы деректер бойынша гипотезаларды тексеру:
   категория бойынша сатылымның температурамен корреляциясы, мерекелер
   маңындағы сатылым, Nager.Date мерекелерін M5-тегі `event_name_1/2`-мен
   салыстыру.

4-тармақ — ең маңыздысы: егер корреляция тіпті визуалды түрде де болмаса,
сыртқы дереккөзді пайплайнға қосуға уақыт жұмсаудың қажеті жоқ.

</details>

<details>
<summary><b>Русский</b></summary>

## `00_defence.ipynb` — презентация проекта (5–7 минут)

Единственный ноутбук, который коммитится **вместе с выводом**: графики и таблицы
в нём и есть то, что показывают. Читает витрину в режиме read-only, ничего не меняет.

Запуск: `uv run jupyter lab notebooks/00_defence.ipynb` либо открыть в VS Code
и выбрать ядро из `.venv`.

Перед показом стоит выполнить `Run All` — числа подтянутся из текущего состояния
витрины, а не останутся зашитыми.

---

## Остальное — только EDA и черновики

Ничего из того, что здесь написано, не является частью пайплайна.

**Правила:**

- Логика, которая пригодилась, переезжает в `src/m5/` и покрывается тестом.
  Ноутбук — это черновик, а не место, где живёт код.
- Вывод ячеек вычищается перед коммитом (`nbstripout` в pre-commit это делает).
- Данные из ноутбуков не сохраняются в `data/processed` — только через CLI.

**Что имеет смысл посмотреть в EDA:**

1. `01_eda_sales.ipynb` — распределение продаж, доля нулей по категориям,
   как выглядят «прерывистые» ряды, когда товары появляются в ассортименте.
2. `02_eda_calendar_snap.ipynb` — эффект SNAP-дней (дни выплат пособий)
   по штатам и категориям. Гипотеза: сильнее всего на FOODS.
3. `03_eda_prices.ipynb` — как часто меняются цены, есть ли видимая реакция
   продаж на снижение цены.
4. `04_eda_external.ipynb` — проверка гипотез по внешним данным:
   корреляция продаж с температурой по категориям, продажи вокруг праздников,
   сверка праздников Nager.Date с `event_name_1/2` из M5.

Пункт 4 — самый важный: если корреляции нет даже визуально, тратить время
на подключение внешнего источника в пайплайн не стоит.

</details>
