"""Общие фикстуры. Модель подменяется на границе провайдера: ``run_agent(agent, prompt)``
в модулях, которые к ней обращаются (как в эталоне курса — подмена функции на уровне модуля).
Тесты с пометкой ``db`` ходят в живую базу из DATABASE_URL и пропускаются без неё."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
import pytest

from app.core import db
from app.core.config import settings

DATA = Path(__file__).resolve().parents[1] / "data"

Reply = str | Exception | Callable[[Any, str], Any]
LLM_MODULES = ("app.services.structured", "app.services.pipeline", "app.services.rag_pipeline")


@dataclass
class FakeLLM:
    """Сценарий ответов модели: по очереди или по имени агента (Extractor, Generator, Critic…)."""

    queue: list[Reply] = field(default_factory=list)
    by_role: dict[str, list[Reply]] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def add(self, *replies: Reply) -> FakeLLM:
        self.queue.extend(replies)
        return self

    def role(self, name: str, *replies: Reply) -> FakeLLM:
        self.by_role.setdefault(name, []).extend(replies)
        return self

    async def __call__(self, agent: Any, prompt: str) -> Any:
        self.calls.append({"role": agent.name, "prompt": prompt, "model": agent.model})
        source = self.by_role.get(agent.name)
        if source:
            reply = source.pop(0) if len(source) > 1 else source[0]
        elif self.queue:
            reply = self.queue.pop(0)
        else:
            raise AssertionError(f"нет ответа для агента {agent.name}")
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(agent, prompt)
        return reply

    def roles(self) -> list[str]:
        return [c["role"] for c in self.calls]


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    fake = FakeLLM()
    for module in LLM_MODULES:
        monkeypatch.setattr(f"{module}.run_agent", fake)
    return fake


def as_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


# --- Живая база --------------------------------------------------------------------------------


def _db_reachable() -> bool:
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2):
            return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def live_db() -> Iterator[None]:
    if not _db_reachable():
        pytest.skip("PostgreSQL недоступен (make infra) — тест на живой базе пропущен")
    from app.core.migrate import migrate

    migrate()
    db.open_pool(wait=True)
    yield
    db.close_pool()


@pytest.fixture
def conn(live_db) -> Iterator[psycopg.Connection]:
    with db.connection() as connection:
        yield connection


@pytest.fixture
def unique() -> str:
    return uuid.uuid4().hex[:8]
