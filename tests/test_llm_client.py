from decimal import Decimal

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from app.agents import prompts
from app.core import db
from app.llm import client
from app.llm.client import LLMResponse


def _status_error(code: int) -> APIStatusError:
    request = httpx.Request("POST", "http://llm/v1/chat/completions")
    return APIStatusError("err", response=httpx.Response(code, request=request), body=None)


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(client.asyncio, "sleep", fake_sleep)
    return sleeps


@pytest.fixture
def recorded(monkeypatch):
    rows: list[dict] = []

    async def fake_record(**fields):
        rows.append(fields)

    monkeypatch.setattr(client, "_record_call", fake_record)
    return rows


def _scripted(monkeypatch, *outcomes):
    calls = []
    queue = list(outcomes)

    async def fake_call(spec, model, prompt, schema):
        calls.append(model)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(client, "_call_model", fake_call)
    return calls


def test_retryable_classification():
    request = httpx.Request("POST", "http://llm")
    assert client._retryable(APIConnectionError(request=request))
    assert client._retryable(TimeoutError())
    assert client._retryable(_status_error(429))
    assert client._retryable(_status_error(503))
    assert not client._retryable(_status_error(400))
    assert not client._retryable(_status_error(401))
    assert not client._retryable(ValueError("что-то своё"))


async def test_retry_on_429_then_success(monkeypatch, no_sleep, recorded):
    calls = _scripted(monkeypatch, _status_error(429), LLMResponse("ok", 10, 5, "m"))
    assert await client.run_agent(prompts.GENERATOR, "привет") == "ok"
    assert len(calls) == 2
    assert len(no_sleep) == 1 and no_sleep[0] >= 1.0
    assert [r["status"] for r in recorded] == ["error", "ok"]


async def test_no_retry_on_bad_request(monkeypatch, no_sleep, recorded):
    calls = _scripted(monkeypatch, _status_error(400))
    with pytest.raises(APIStatusError):
        await client.run_agent(prompts.GENERATOR, "привет")
    assert len(calls) == 1
    assert no_sleep == []


async def test_gives_up_after_max_retries(monkeypatch, no_sleep, recorded):
    request = httpx.Request("POST", "http://llm")
    errors = [APIConnectionError(request=request) for _ in range(3)]
    calls = _scripted(monkeypatch, *errors)
    with pytest.raises(APIConnectionError):
        await client.run_agent(prompts.GENERATOR, "привет")
    assert len(calls) == 3
    assert len(no_sleep) == 2
    assert no_sleep[1] > no_sleep[0] * 1.2  # задержка растёт


async def test_cost_recorded_and_cheap_model_for_service_roles(monkeypatch, recorded):
    calls = _scripted(
        monkeypatch, LLMResponse("{}", 1000, 500, "cheap"), LLMResponse("{}", 1000, 500, "main")
    )
    await client.run_agent(prompts.CRITIC, "проверь")
    await client.run_agent(prompts.GENERATOR, "сгенерируй")
    assert calls == [client.settings.llm_cheap_model, client.settings.llm_model]
    cheap, main = recorded
    assert (cheap["tier"], main["tier"]) == ("cheap", "main")
    assert isinstance(main["cost"], Decimal)
    assert main["cost"] > cheap["cost"] > 0
    assert main["input_tokens"] == 1000 and main["output_tokens"] == 500


async def test_job_id_from_log_context_goes_to_call(monkeypatch, recorded):
    from app.core.logging import bind_context, clear_context

    _scripted(monkeypatch, LLMResponse("{}", 1, 1, "m"))
    clear_context()
    bind_context(job_id="job123", request_id="req1")
    try:
        await client.run_agent(prompts.GENERATOR, "x", purpose="generate")
    finally:
        clear_context()
    assert recorded[0]["job_id"] == "job123"
    assert recorded[0]["request_id"] == "req1"
    assert recorded[0]["purpose"] == "generate"


async def test_recording_skipped_without_pool(monkeypatch):
    monkeypatch.setattr(db, "_pool", None)
    _scripted(monkeypatch, LLMResponse("{}", 1, 1, "m"))
    assert await client.run_agent(prompts.GENERATOR, "x") == "{}"
