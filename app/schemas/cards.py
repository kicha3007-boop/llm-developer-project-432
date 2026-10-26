"""Контракт результата: черновик карточки и промежуточные результаты агентов.

Имена и поля — как в эталоне курса (урок «Разбор примера»: ``SupplierFacts``, ``CardDraft``,
``CritiqueReport``), расширенные шагом «Контракт результата»: seo-блок, ссылки на источники,
недостающие поля и уверенность. Приложение проверяет ответ модели этими моделями так же,
как ответ своей базы: до того, как что-то с ним сделать.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TITLE_MAX_LENGTH = 60
_EMPTY_VALUES = {"-", "—", "нет данных", "n/a", "null", "none", "неизвестно"}


def _norm_key(name: str) -> str:
    return " ".join(name.strip().lower().replace("ё", "е").split())


def _drop_empty_values(value: Any) -> Any:
    """Характеристика с пустым значением в карточку не попадает."""
    if not isinstance(value, dict):
        return value
    cleaned: dict[str, str] = {}
    for key, raw in value.items():
        if raw is None:
            continue
        text = str(raw).strip()
        if not text or text.lower() in _EMPTY_VALUES:
            continue
        cleaned[str(key).strip()] = text
    return cleaned


class SourceRef(BaseModel):
    """Ссылка на фрагмент. ``chunk_id`` (и ``field``) указывает модель; ``doc_id``, ``page``,
    ``section`` и ``score`` заполняет система из базы и результатов поиска."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(min_length=1)
    doc_id: str = ""
    page: int | None = None
    score: float | None = None
    field: str | None = None
    section: str | None = None


class Seo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: str = ""
    description: str = ""
    keywords: list[str] = Field(default_factory=list)


class CardDraft(BaseModel):
    """Черновик карточки товара — итоговый контракт сервиса."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1, max_length=TITLE_MAX_LENGTH)
    description: str = ""
    characteristics: dict[str, str] = Field(default_factory=dict)
    benefits: list[str] = Field(default_factory=list)
    seo: Seo = Field(default_factory=Seo)
    sources: list[SourceRef] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    # Если модель уверенность не указала, её ограничивает код (штраф за недостающие поля)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("title", mode="before")
    @classmethod
    def _strip_title(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("characteristics", mode="before")
    @classmethod
    def _drop_empty(cls, value: Any) -> Any:
        return _drop_empty_values(value)

    @field_validator("missing_fields")
    @classmethod
    def _dedupe_missing(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        result = []
        for item in value:
            item = str(item).strip()
            if item and _norm_key(item) not in seen:
                seen.add(_norm_key(item))
                result.append(item)
        return result

    @model_validator(mode="after")
    def _field_not_in_both_lists(self) -> CardDraft:
        present = {_norm_key(k) for k in self.characteristics}
        both = [m for m in self.missing_fields if _norm_key(m) in present]
        if both:
            raise ValueError(
                "поле не может быть одновременно в characteristics и в missing_fields: "
                + ", ".join(both)
            )
        return self


class SupplierFacts(BaseModel):
    """Результат извлекателя: структурированные факты из текста поставщика."""

    model_config = ConfigDict(extra="ignore")

    product_name: str = ""
    characteristics: dict[str, str] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)

    @field_validator("characteristics", mode="before")
    @classmethod
    def _drop_empty(cls, value: Any) -> Any:
        return _drop_empty_values(value)


class CritiqueReport(BaseModel):
    """Вердикт критика. Literal превращается в enum JSON-схемы."""

    model_config = ConfigDict(extra="ignore")

    verdict: Literal["approve", "regenerate"]
    issues: list[str] = Field(default_factory=list)

    @field_validator("issues", mode="before")
    @classmethod
    def _issues_to_strings(cls, value: Any) -> Any:
        # модели иногда возвращают замечания объектами {"field": ..., "message": ...}
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, dict):
                    msg = str(item.get("message") or item.get("issue") or item)
                    field = item.get("field")
                    out.append(f"{field}: {msg}" if field else msg)
                else:
                    out.append(str(item))
            return out
        return value


class InjectionAnswer(BaseModel):
    """Ответ дешёвой модели-детектора инъекций (разбирается в ``InjectionVerdict``)."""

    model_config = ConfigDict(extra="ignore")

    is_injection: bool
    reason: str = ""


class JudgeVerdict(BaseModel):
    """Вердикт модели-судьи: подтверждаются ли утверждения карточки источниками."""

    model_config = ConfigDict(extra="ignore")

    supported: int = Field(ge=0)
    total: int = Field(ge=0)
    unsupported_claims: list[str] = Field(default_factory=list)

    @property
    def score(self) -> float:
        return self.supported / self.total if self.total else 0.0


# --- HTTP-контракты ---------------------------------------------------------------------------


class JobIn(BaseModel):
    supplier_text: str = Field(min_length=1)


class CardResponse(BaseModel):
    status: str
    attempts: int
    verdict: str
    card: CardDraft | None = None


class GenerateCardRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1)
    product_hint: str | None = None
