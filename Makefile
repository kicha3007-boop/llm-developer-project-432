.PHONY: setup install infra infra-down infra-reset migrate api worker reindex search lint format test test-db check evals evals-all stuck cost

UV ?= uv
RUN = $(UV) run --frozen

# Подготовка с нуля: зависимости и, если задан DATABASE_URL, миграции (её вызывает автопроверка Хекслета)
setup: install
	@if [ -n "$$DATABASE_URL" ]; then $(RUN) python -m app.core.migrate; fi

# Зависимости — из файла блокировки, одной командой
install:
	$(UV) sync --frozen

# База с pgvector и сервер Temporal (веб-интерфейс http://localhost:8233)
infra:
	docker compose up -d --wait

infra-down:
	docker compose down

# Полный сброс состояния: контейнеры и тома
infra-reset:
	docker compose down -v

migrate:
	$(RUN) python -m app.core.migrate

api:
	$(RUN) uvicorn app.main:app --host 0.0.0.0 --port 8000

# Воркер процессов Temporal: генерация по тексту, разбор документов, генерация по документам
worker:
	$(RUN) python -m app.temporal.worker

# Досчёт векторов фрагментам без эмбеддинга (повторный запуск — пустая работа)
reindex:
	$(RUN) python -m app.rag.reindex

# make search Q="мотор 800 Вт" MODE=hybrid
search:
	$(RUN) python -m app.rag.retrieval "$(Q)" --mode $(or $(MODE),hybrid)

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

# Тесты не требуют модели и сети; тесты с пометкой db пропускаются, если база недоступна
test:
	$(RUN) pytest -q

test-db:
	$(RUN) pytest -q -m db

check: lint test

# Прогон метрик на эталоне (по умолчанию подмножество eval_defaults; ALL=1 — весь набор)
evals:
	$(RUN) python -m evals.run_evals $(if $(ALL),--all,)

evals-all:
	$(RUN) python -m evals.run_evals --all

# Задачи, зависшие в нетерминальном состоянии
stuck:
	docker compose exec -T db psql -U card -d card -c "SELECT id, status, attempts, updated_at FROM jobs WHERE status NOT IN ('confirmed','rejected','failed','approved','needs_review') AND updated_at < now() - interval '15 minutes';"

# make cost JOB=<job_id> — стоимость одной карточки одним запросом
cost:
	docker compose exec -T db psql -U card -d card -c "SELECT tier, model, count(*) AS calls, sum(cost) AS cost, sum(input_tokens) AS in_tok, sum(output_tokens) AS out_tok FROM llm_calls WHERE job_id = '$(JOB)' GROUP BY ROLLUP (tier, model);"
