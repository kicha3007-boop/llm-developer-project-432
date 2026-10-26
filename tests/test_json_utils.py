import pytest

from app.services.json_utils import LLMJsonError, parse_llm_json


def test_clean_json():
    assert parse_llm_json('{"title": "Блендер", "confidence": 0.9}') == {
        "title": "Блендер",
        "confidence": 0.9,
    }


def test_json_in_markdown_fence():
    raw = 'Вот карточка:\n```json\n{"title": "Чайник", "benefits": ["быстро"]}\n```\n'
    assert parse_llm_json(raw) == {"title": "Чайник", "benefits": ["быстро"]}


def test_json_with_text_before_and_after():
    raw = 'Конечно! Результат: {"verdict": "approve", "issues": []} Надеюсь, помог.'
    assert parse_llm_json(raw) == {"verdict": "approve", "issues": []}


def test_nested_brackets_inside_strings():
    raw = 'Ответ: {"a": {"b": [1, 2]}, "c": "скобка } внутри"} конец'
    assert parse_llm_json(raw) == {"a": {"b": [1, 2]}, "c": "скобка } внутри"}


def test_json_after_reasoning_block():
    assert parse_llm_json('<think>подумаю {не json}</think>\n{"a": 1}') == {"a": 1}


def test_garbage_gives_clear_error():
    with pytest.raises(LLMJsonError, match="нет JSON"):
        parse_llm_json("Извините, я не могу помочь с этим запросом.")


def test_unclosed_json():
    with pytest.raises(LLMJsonError, match="не закрыт"):
        parse_llm_json('{"title": "Блендер"')


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_empty_response_is_error(raw):
    with pytest.raises(LLMJsonError, match="пустой ответ"):
        parse_llm_json(raw)
