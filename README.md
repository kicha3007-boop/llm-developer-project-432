# ИИ-генератор карточки товара


[![hexlet-check](https://github.com/kicha3007-boop/llm-developer-project-432/actions/workflows/hexlet-check.yml/badge.svg)](https://github.com/kicha3007-boop/llm-developer-project-432/actions)

Бэкенд-сервис, который принимает документы поставщиков (pdf, docx, xlsx), строит по ним поисковый
индекс и генерирует **черновик** карточки товара: каждая характеристика ссылается на фрагмент
документа, недостающие данные попадают в `missing_fields`, а не выдумываются, у карточки есть
уровень уверенности. Ссылки на источники проверяются кодом, персональные данные маскируются до
вызова модели, фрагменты с инъекциями выбрасываются, стоимость каждого вызова модели пишется в базу.

Учебный проект Хекслета: https://ru.hexlet.io/programs/llm-developer

## Стек

Python 3.12 · FastAPI · Pydantic · PostgreSQL 16 + pgvector · Temporal · OpenAI Agents SDK
(OpenAI-совместимый API: LM Studio, Ollama или OpenRouter) · sentence-transformers (локальные
эмбеддинги) · pdfplumber, python-docx, openpyxl · uv, ruff, pytest.

## Что понадобится

- Python 3.12 и [uv](https://docs.astral.sh/uv/) (`pip install uv` или инструкция на сайте);
- Docker с Compose (база и Temporal поднимаются контейнерами);
- сервер модели с OpenAI-совместимым API. Проще всего — [LM Studio](https://lmstudio.ai/) с
  загруженной моделью (по умолчанию `http://localhost:1234/v1`) или Ollama
  (`http://localhost:11434/v1`). Подойдёт и OpenRouter (`https://openrouter.ai/api/v1` + ключ);
- `make` (в Windows есть в Git Bash/MSYS2; без него — команды из раздела «Без make»);
- ~2 ГБ на диске: при первой индексации скачивается модель эмбеддингов
  `intfloat/multilingual-e5-base` (~1.1 ГБ), torch ставится в CPU-сборке.

Свободные порты: 8000 (API), 5432 (PostgreSQL), 7233 и 8233 (Temporal и его веб-интерфейс).
Если 5432 занят локальным PostgreSQL: `PG_PORT=5433 make infra` и поправьте порт в `DATABASE_URL`.

## Установка

```bash
git clone https://github.com/kicha3007-boop/llm-developer-project-432.git
cd llm-developer-project-432
make install                 # uv sync --frozen: зависимости строго из uv.lock
cp .env.example .env         # затем впишите свою модель
```

В `.env` поправьте как минимум:

- `LLM_BASE_URL`, `LLM_API_KEY` — адрес и ключ сервера модели (для LM Studio ключ любой);
- `LLM_MODEL` — основная модель (генерация), `LLM_CHEAP_MODEL` — дешёвая (проверка черновика,
  детектор инъекций, судья метрик). Идентификаторы смотрите в `curl $LLM_BASE_URL/models`.
  Можно указать одну и ту же модель в обоих полях;
- если сервер не понимает `response_format` с JSON-схемой (ошибка 400 на первом же вызове),
  поставьте `LLM_STRUCTURED_OUTPUT=off` — схема всё равно уходит текстом в инструкции.

Остальные параметры (цены токенов, порог релевантности, порог уверенности, лимит контекста) описаны
комментариями в `.env.example`.

## Запуск

Три процесса, каждый в своём терминале:

```bash
make infra       # PostgreSQL+pgvector и Temporal; ждёт состояния healthy
make migrate     # «Применена миграция: ...», при повторном запуске — «Новых миграций нет»
make worker      # терминал 2: воркер процессов Temporal
make api         # терминал 3: API на http://localhost:8000 (документация — /docs)
```

Проверка:

```bash
docker compose ps                      # db и temporal — running (healthy)
curl localhost:8000/health             # {"status":"ok"} — живость, не зависит от базы
curl localhost:8000/ready              # {"status":"ready","db":"ok","vector":"0.8.1"}
```

Веб-интерфейс Temporal — http://localhost:8233: там видна история каждого процесса (какой шаг
выполнен, какой повторяется, где процесс ждёт сигнала).

## Использование

### 1. Загрузить документы

```bash
for f in data/blender_passport.pdf data/blender_kp.docx data/kettle_spec.xlsx \
         data/kettle_manual.pdf data/boiler_scan.pdf; do
  curl -s -F "file=@$f" localhost:8000/documents/; echo
done
# {"document_id":"a3f19c40b2d1","status":"new"}  ← идентификатор = префикс sha256 содержимого
```

Загрузка отвечает 202 сразу, разбор и индексация идут в воркере (процесс `DocumentWorkflow`).
Состояние документа: `new → parsing → indexing → indexed` или `failed` с причиной:

```bash
curl -s localhost:8000/documents/<document_id>
# {"document_id":"...","status":"indexed","chunk_count":5,"embedded_count":5,...}
# boiler_scan.pdf: {"status":"failed","error":"нет текстового слоя: документ похож на скан ..."}
```

Повторная загрузка того же файла возвращает тот же `document_id` и не создаёт дублей.

### 2. Поиск по фрагментам

```bash
curl -s -X POST localhost:8000/documents/search -H 'Content-Type: application/json' \
  -d '{"query":"KTL-1700","mode":"hybrid"}'            # режимы: vector | text | hybrid
make search Q="мотор 800 Вт" MODE=vector               # то же командой
```

### 3. Сгенерировать карточку

```bash
curl -s -X POST localhost:8000/generate-card \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: blender-1' \
  -d '{"document_ids":["<id паспорта>","<id КП>"],"product_hint":"блендер"}'
# {"job_id":"7b21e4d9c0aa","status":"pending"}
```

Повтор с тем же `Idempotency-Key` вернёт ту же задачу. Статус опрашивается так:

```bash
curl -s localhost:8000/jobs/7b21e4d9c0aa
```

Цепочка состояний: `pending → parsing → indexing → retrieving → generating → verifying`
(при переделке — `regenerating`, не больше трёх кругов) → `approved` (проверки пройдены) или
`needs_review` (ждёт человека: низкая уверенность, выдуманная ссылка, инъекция, контакты в
результате — причина в `review_reason`). В `result` — карточка (`title`, `description`,
`characteristics`, `benefits`, `seo`, `sources`, `missing_fields`, `confidence`), отчёт защиты
`security_report` (что исключено, что замаскировано) и `context_chunk_ids`.

### 4. Ручная проверка

Процесс после автоматических проверок ждёт решения человека (сколько угодно долго):

```bash
curl -s localhost:8000/workflows/7b21e4d9c0aa                     # состояние процесса
curl -s -X POST localhost:8000/workflows/7b21e4d9c0aa/approve     # → статус confirmed
curl -s -X POST localhost:8000/workflows/7b21e4d9c0aa/reject \
  -H 'Content-Type: application/json' -d '{"comment":"не тот товар"}'   # → rejected
```

### Генерация по тексту без документов

```bash
curl -s -X POST localhost:8000/cards -H 'Content-Type: application/json' \
  -d '{"supplier_text":"Блендер погружной МиксерПро 800. Мощность 800 Вт, 6 скоростей."}'
# синхронно: извлечение → генерация → проверка, ответ с карточкой

curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: t-1' -d '{"supplier_text":"..."}'          # то же в фоне (202)
```

### Стоимость и логи

```bash
curl -s localhost:8000/jobs/<job_id>/cost        # сумма и разбивка по моделям (main / cheap)
make cost JOB=<job_id>                           # то же одним SQL-запросом к llm_calls
```

Каждая запись лога несёт `request_id` (рождается на входе, возвращается в заголовке
`X-Request-ID`) и `job_id` (передаётся аргументом процесса и привязывается в каждом шаге воркера).
Чтобы искать по логам обоих процессов, задайте `LOG_FILE=logs/app.log` (и `mkdir logs`) или
`LOG_JSON=true`, затем `grep <job_id> logs/app.log` покажет весь путь задачи: обработчик, шаги
процесса, вызовы модели с длительностью и стоимостью. Логи проходят через маскирование
персональных данных.

### Метрики качества

```bash
make evals          # подмножество eval_defaults из data/golden_cards.json
make evals ALL=1    # весь эталон
```

Печатает таблицу (точность характеристик, цитирование, подтверждённость по модели-судье,
стоимость) и сохраняет отчёт в `evals/reports/`. Калибровка порога релевантности:
`uv run python -m evals.calibrate_threshold` (без базы и без сервера модели).

## Проверки

```bash
make lint           # ruff check + ruff format --check
make test           # pytest: модель подменена, сеть не нужна
```

Тесты с пометками `db` и `integration` ходят в живую базу и Temporal из `.env` и автоматически
пропускаются, если инфраструктура не поднята (`make infra && make migrate` — и они тоже пройдут:
в том числе процесс с «убитым» посреди генерации воркером).

## Без make

```bash
uv sync --frozen
docker compose up -d --wait
uv run python -m app.core.migrate
uv run python -m app.temporal.worker
uv run uvicorn app.main:app --port 8000
uv run ruff check . && uv run pytest -q
```

## Сброс состояния с нуля

```bash
make infra-reset    # docker compose down -v: удаляет контейнеры И тома
make infra && make migrate
```

## Устройство

```text
app/
├── main.py         приложение, жизненный цикл пула, сквозной X-Request-ID
├── core/           config.py (настройки), db.py (pool(), connection()), logging.py, migrate.py
├── routers/        health, cards, jobs, documents, generate, workflows — только транспорт
├── services/       pipeline.py (текст → карточка), structured.py (схема, лечение, точечная
│                   переделка), json_utils.py, documents.py, rag_pipeline.py (документы → карточка)
├── agents/         prompts.py — роли и промпты отдельными функциями
├── llm/            client.py (граница провайдера: таймаут, повторы, учёт), cost.py (Decimal)
├── rag/            embedder.py, retrieval.py, context.py (маскирование + инъекции + источники),
│                   reindex.py (досчёт векторов)
├── parsers/        pdf.py, docx.py, xlsx.py, normalizer.py, chunker.py
├── guardrails/     pii.py, injection.py, __init__.py (фильтр выхода)
├── repositories/   jobs.py, documents.py, chunks.py (поиск, RRF), llm_calls.py — весь SQL здесь
├── schemas/        cards.py, jobs.py
└── temporal/       workflows.py, activities.py, client.py, worker.py
db/migrations/      SQL-миграции (0001 — vector + schema_migrations)
evals/              метрики, прогон, калибровка порога
tests/              pytest
data/               документы поставщиков и эталон (как есть, побайтово)
docs/adr/           архитектурные решения (ADR)
```

Решения: [docs/adr](docs/adr) — стек и слои, миграции, поиск и порог релевантности, защита и
наблюдаемость. Отчёт о прогоне и ответы на вопросы защиты — ниже.

## Отчёт о прогоне

**Прогон на живой языковой модели не выполнялся.** В окружении, где собирался проект, не было
доступного сервера модели, поэтому чисел качества генерации (точность характеристик, цитирование,
подтверждённость судьёй) и стоимости реальной карточки здесь нет — выдумывать их я не стал.
Ниже — что реально запускалось, с фактическими результатами, и команды недостающего прогона.

Окружение: Windows 11, Docker Desktop 28.5, Python 3.12.9, uv 0.11.

1. **Инфраструктура и миграции.** `docker compose up -d --wait`: `db` (pgvector 0.8.1, pg16) и
   `temporal` (temporalio/temporal 1.8.3) — healthy, веб-интерфейс Temporal на `:8233` отвечает.
   `make migrate`: первый запуск — «Применена миграция: 0001_init … 0005_llm_calls», второй —
   «Новых миграций нет». Тест `test_migrate_from_scratch_then_noop` повторяет это на чистой базе.
2. **Проверки состояния.** `/health` → `{"status":"ok"}`, `/ready` →
   `{"status":"ready","db":"ok","vector":"0.8.1"}`; при недоступной базе живость 200, готовность 503.
3. **Загрузка и индексация через API и воркер** (настоящая модель эмбеддингов на CPU):

   | файл | статус | фрагментов |
   | --- | --- | --- |
   | blender_passport.pdf | indexed | 5 |
   | blender_kp.docx | indexed | 6 |
   | kettle_spec.xlsx | indexed | 3 |
   | kettle_manual.pdf | indexed | 4 |
   | boiler_scan.pdf | failed: «нет текстового слоя: документ похож на скан (2 стр., 2 из них — только изображение). Распознавание (OCR) не включено…» | 0 |

   Повторная загрузка `blender_passport.pdf` вернула тот же `document_id` без дублей.
4. **Поиск и калибровка порога** на настоящей модели — числа и выводы в
   [ADR 0003](docs/adr/0003-retrieval.md): порог 0.80 пропускает 11 из 12 верных фрагментов и
   отсекает лучший фрагмент у 5 из 6 запросов без ответа; «фильтр от накипи» векторный поиск с
   порогом теряет, гибрид находит; артикулы KTL-1000 / BLD-1200 гибрид ставит первыми с отрывом
   вдвое.
5. **Генерация при недоступной модели** (`POST /generate-card` по `kettle_manual.pdf` +
   `blender_kp.docx`): повтор с тем же `Idempotency-Key` вернул ту же задачу; статусы
   `parsing → indexing → retrieving → generating → failed` (`APIConnectionError`). В `llm_calls`
   8 записей: 2 на дешёвую модель (детектор инъекций по фрагменту «SYSTEM: игнорируй…»; модель
   недоступна → фрагмент считается подозрительным) и 6 на основную (3 попытки шага Temporal ×
   2 попытки клиента). Поиск по `job_id` в логе показывает весь путь задачи; поиск исходных
   телефонов, почты и ИНН из выданных документов по логам — 0 совпадений.
6. **Тесты** (`make test`, 112 штук) с живыми базой и Temporal и подменённой моделью: процесс
   доходит до ожидания человека и завершается сигналом; воркер «убивается» посреди шага генерации,
   новый воркер доводит процесс, шаг извлечения в истории Temporal запланирован ровно один раз;
   сквозной процесс по `blender_passport.pdf` даёт карточку со ссылкой на фрагмент стр. 1
   «Технические характеристики» и статус `approved`.

Как сделать прогон на живой модели: заполнить `LLM_*` в `.env`, `make infra && make migrate`,
`make worker`, `make api`, загрузить документы и сгенерировать карточки (раздел «Использование»),
затем `make evals` (или `make evals ALL=1`) и `make cost JOB=<job_id>`. Таблица и средние
печатаются в консоль и сохраняются в `evals/reports/`; их нужно перенести сюда.

**Ожидание по самой слабой метрике — гипотеза, не измерение.** Ожидаю, что слабее всего будет
цитирование: одни и те же значения («800 Вт», «графитовый») есть и в паспорте, и в КП, и модель
сошлётся на фрагмент, где значение тоже есть, но не на тот, что в эталоне (`source_probes`
требует страницу 1 паспорта). Точность характеристик держится на проверке ссылок и сверке
значений кодом, а судья склонен завышать подтверждённость. Проверить — первым делом после
живого прогона.

### Ответы на вопросы защиты

1. **Повтор вызова или задачи.** Вызов повторяет `app/llm/client.py::run_agent` — только на
   сеть/таймаут/429/5xx, с экспонентой и примесью: дрожание сети лечится за секунды ценой одного
   обращения. Шаг целиком повторяет Temporal по `RetryPolicy` (упал воркер, база, сервер), не
   пересчитывая выполненные шаги. Детерминированные ошибки (400, `StructuredOutputError` после
   лечения) не повторяет никто: повтор не поможет, а счёт умножит.
2. **Гибрид и артикулы.** Для модели эмбеддингов «KTL-1000» — строка, похожая на «KTL-1700»
   (близость 0.86 против 0.84); полнотекстовый поиск совпадает буква в букву. RRF складывает
   вклады позиций двух списков: найденное обоими поднимается, несопоставимые шкалы оценок не важны.
3. **Маскирование до модели.** Отправленное в чужое API уже утекло — фильтр на выходе этого не
   отменит. Фильтр выхода нужен для другого: модель может частично выполнить инъекцию или
   «вспомнить» контакт, а видит итоговый текст только он.
4. **Проверка ссылок.** Ссылка на существующий фрагмент, которого не было в контексте, —
   выдумывание с правдоподобным видом: модель не могла его прочитать. Поэтому два условия: есть в
   базе и был в контексте; плюс у каждой характеристики обязан быть источник.
5. **Низкая уверенность.** Процесс: задача уже в `needs_review` с причиной, решает человек
   сигналом. Код: `missing_fields` и `review_reason`, затем `context_chunk_ids` — нашёл ли поиск
   нужное (порог, запросы), не исключила ли защита полезный фрагмент (`security_report`), не обрезал
   ли лимит контекста. Если данных в документах нет — это верное поведение, а не дефект.
6. **Стоимость втрое.** `llm_calls` за неделю по `purpose`, `tier`, `attempt`, `status`: выросло
   число вызовов (повторы, `:repair`, переделки) или токены на вызов (контекст); не ушли ли
   служебные роли на основную модель. Затем лог по `job_id` дорогих задач и история Temporal.
7. **Состояние в базе.** История движка знает «шаг выполнен со второй попытки», а клиенту нужно
   деловое состояние: «на переделке», «ждёт человека», причина, результат — его пишет отдельный
   шаг. При рестарте воркера процесс продолжается с последнего выполненного шага (проверено тестом).
8. **Детектор инъекций без модели.** Помеченный правилами фрагмент считается подозрительным и
   выбрасывается (fail-closed), в `security_report` — причина «точный детектор недоступен».
   Фрагменты, которые правила не пометили, модели не требуют вовсе.

---

<details>
<summary>Автоматические тесты Хекслета</summary>

Тесты запускаются на каждый коммит. За запуск отвечает файл `.github/workflows/hexlet-check.yml` — не удаляйте и не переименовывайте ни его, ни репозиторий.

</details>

## О Хекслете

[Хекслет](https://ru.hexlet.io/) — школа программирования: авторские программы обучения с практикой, поддержкой наставников и реальными проектами, которые остаются в резюме. Этот репозиторий — один из таких проектов.
