"""Общие фикстуры. Модель подменяется на границе провайдера — ``app.llm.client.run_agent``
(или глубже, ``_call_model``, в тестах самого клиента). Тесты с пометкой ``db`` ходят в живую
базу из DATABASE_URL и пропускаются, если она недоступна."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
import pytest

from app.agents.prompts import AgentSpec
from app.core import db
from app.core.config import settings
from app.llm import client as llm_client

DATA = Path(__file__).resolve().parents[1] / "data"

Reply = str | Exception | Callable[[AgentSpec, str], str]


@dataclass
class FakeLLM:
    """Сценарий ответов модели: по очереди или по имени роли."""

    queue: list[Reply] = field(default_factory=list)
    by_role: dict[str, list[Reply]] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def add(self, *replies: Reply) -> FakeLLM:
        self.queue.extend(replies)
        return self

    def role(self, name: str, *replies: Reply) -> FakeLLM:
        self.by_role.setdefault(name, []).extend(replies)
        return self

    async def __call__(
        self,
        spec: AgentSpec,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        purpose: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "role": spec.name,
                "prompt": prompt,
                "purpose": purpose,
                "cheap": spec.cheap,
                "schema": schema,
            }
        )
        source = self.by_role.get(spec.name)
        if source:
            reply = source.pop(0) if len(source) > 1 else source[0]
        elif self.queue:
            reply = self.queue.pop(0)
        else:
            raise AssertionError(f"нет ответа для роли {spec.name}")
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(spec, prompt)
        return reply

    def roles(self) -> list[str]:
        return [c["role"] for c in self.calls]


@pytest.fixture
def fake_llm(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    fake = FakeLLM()
    monkeypatch.setattr(llm_client, "run_agent", fake)
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
def unique() -> str:
    return uuid.uuid4().hex[:8]
