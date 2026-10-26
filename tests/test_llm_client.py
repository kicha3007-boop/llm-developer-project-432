"""Клиент модели: повторы только на повторяемых ошибках, учёт каждого вызова, две модели.

Подменяется исполнитель SDK (``Runner``) и функция сна — как в эталоне курса."""

from decimal import Decimal

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from app.agents.prompts import critic_agent, generator_agent
from app.core import db
from app.llm import client


def _status_error(code: int) -> APIStatusError:
    request = httpx.Request("POST", "http://llm/v1/chat/completions")
    return APIStatusError("err", response=httpx.Response(code, request=request), body=None)


class _Usage:
    def __init__(self, i: int, o: int) -> None:
        self.input_tokens, self.output_tokens = i, o


class _Result:
    def __init__(self, output: str = "ok", i: int = 0, o: int = 0) -> None:
        self.final_output = output
        self.context_wrapper = type("W", (), {"usage": _Usage(i, o)})()


@pytest.fixture
def runner(monkeypatch):
    """Сценарий ответов исполнителя SDK и записанные паузы."""
    state: dict = {"queue": [], "calls": [], "sleeps": []}

    class FakeRunner:
        @staticmethod
        async def run(agent, prompt):
            state["calls"].append(agent.name)
            item = state["queue"].pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    async def fake_sleep(delay):
        state["sleeps"].append(delay)

    monkeypatch.setattr(client, "setup_llm", lambda: None)
    monkeypatch.setattr(client, "Runner", FakeRunner)
    monkeypatch.setattr(client.asyncio, "sleep", fake_sleep)
    return state


@pytest.fixture
def recorded(monkeypatch):
    rows: list[dict] = []
    monkeypatch.setattr(client, "_record", rows.append)
    return rows


def test_retryable_classification():
    request = httpx.Request("POST", "http://llm")
    assert client._retryable(APIConnectionError(request=request))
    assert client._retryable(TimeoutError())
    assert client._retryable(_status_error(429))
    assert client._retryable(_status_error(503))
    assert not client._retryable(_status_error(400))
    assert not client._retryable(_status_error(401))
    assert not client._retryable(ValueError("своя ошибка"))


async def test_retry_on_429_then_success(runner, recorded):
    runner["queue"] += [_status_error(429), _Result("ok", 10, 5)]
    assert await client.run_agent(generator_agent, "привет") == "ok"
    assert runner["calls"] == ["Generator", "Generator"]
    assert len(runner["sleeps"]) == 1 and runner["sleeps"][0] >= 1.0
    assert [r["status"] for r in recorded] == ["error", "ok"]
    assert recorded[1]["attempt"] == 2


async def test_no_retry_on_bad_request(runner, recorded):
    runner["queue"] += [_status_error(400)]
    with pytest.raises(APIStatusError):
        await client.run_agent(generator_agent, "привет")
    assert len(runner["calls"]) == 1
    assert runner["sleeps"] == []


async def test_gives_up_after_max_retries(runner, recorded):
    request = httpx.Request("POST", "http://llm")
    runner["queue"] += [APIConnectionError(request=request) for _ in range(3)]
    with pytest.raises(APIConnectionError):
        await client.run_agent(generator_agent, "привет")
    assert len(runner["calls"]) == 3
    assert len(runner["sleeps"]) == 2
    assert runner["sleeps"][1] > runner["sleeps"][0] * 1.2  # задержка растёт


async def test_cost_recorded_and_cheap_model_for_service_roles(runner, recorded):
    runner["queue"] += [_Result("{}", 1000, 500), _Result("{}", 1000, 500)]
    await client.run_agent(critic_agent, "проверь")
    await client.run_agent(generator_agent, "сгенерируй")
    cheap, main = recorded
    assert (cheap["tier"], main["tier"]) == ("cheap", "main")
    assert cheap["model"] == client.settings.llm_cheap_model
    assert isinstance(main["cost"], Decimal)
    assert main["cost"] > cheap["cost"] > 0
    assert (main["input_tokens"], main["output_tokens"]) == (1000, 500)
    assert main["purpose"] == "Generator"


async def test_job_id_from_log_context_goes_to_call(runner, recorded):
    from app.core.logging import bind_context, clear_context

    runner["queue"] += [_Result()]
    clear_context()
    bind_context(job_id="job123", request_id="req1")
    try:
        await client.run_agent(generator_agent, "x")
    finally:
        clear_context()
    assert (recorded[0]["job_id"], recorded[0]["request_id"]) == ("job123", "req1")


async def test_agent_without_usage_is_fine(runner, monkeypatch):
    class Bare:
        final_output = "ok"

    class FakeAgent:
        name = "test"

    monkeypatch.setattr(db, "_pool", None)  # без пула учёт пропускается
    runner["queue"] += [Bare()]
    assert await client.run_agent(FakeAgent(), "x") == "ok"
