import pytest

from app.services.json_utils import JSONExtractionError, extract_json, extract_json_object


def test_clean_json():
    assert extract_json('{"title": "Блендер", "confidence": 0.9}') == {
        "title": "Блендер",
        "confidence": 0.9,
    }


def test_json_in_markdown_fence():
    raw = 'Вот карточка:\n```json\n{"title": "Чайник", "benefits": ["быстро"]}\n```\n'
    assert extract_json(raw) == {"title": "Чайник", "benefits": ["быстро"]}


def test_json_with_text_before_and_after():
    raw = 'Конечно! Результат: {"verdict": "approve", "issues": []} Надеюсь, помог.'
    assert extract_json(raw) == {"verdict": "approve", "issues": []}


def test_json_after_reasoning_block():
    raw = '<think>подумаю {не json}</think>\n{"a": 1}'
    assert extract_json(raw) == {"a": 1}


def test_garbage_gives_clear_error():
    with pytest.raises(JSONExtractionError, match="не является JSON"):
        extract_json("Извините, я не могу помочь с этим запросом.")


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_empty_response_is_error(raw):
    with pytest.raises(JSONExtractionError, match="пустой ответ"):
        extract_json(raw)


def test_object_expected():
    with pytest.raises(JSONExtractionError, match="ожидался JSON-объект"):
        extract_json_object("[1, 2, 3]")
