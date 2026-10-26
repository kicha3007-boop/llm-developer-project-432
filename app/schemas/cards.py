"""Контракт результата: карточка товара и промежуточные результаты агентов.

Приложение проверяет ответ модели этими моделями так же, как ответ своей базы:
до того, как что-то с ним сделать.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.core.config import settings

TITLE_MAX_LEN = settings.title_max_len
SEO_TITLE_MAX_LEN = 70
SEO_DESCRIPTION_MAX_LEN = 160


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
        if not text or text.lower() in {
            "-",
            "—",
            "нет данных",
            "n/a",
            "null",
            "none",
            "неизвестно",
        }:
            continue
        cleaned[str(key).strip()] = text
    return cleaned


class SourceRef(BaseModel):
    """Ссылка на фрагмент документа, из которого взято утверждение."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(min_length=1)
    field: str | None = None
    page: int | None = None
    section: str | None = None


class SeoBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")

    meta_title: str = Field(default="", max_length=SEO_TITLE_MAX_LEN)
    meta_description: str = Field(default="", max_length=SEO_DESCRIPTION_MAX_LEN)
    keywords: list[str] = Field(default_factory=list)


class ProductCard(BaseModel):
    """Черновик карточки товара — итоговый контракт сервиса."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1, max_length=TITLE_MAX_LEN)
    description: str = ""
    characteristics: dict[str, str] = Field(default_factory=dict)
    benefits: list[str] = Field(default_factory=list)
    seo: SeoBlock = Field(default_factory=SeoBlock)
    sources: list[SourceRef] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("characteristics", mode="before")
    @classmethod
    def _drop_empty(cls, value: Any) -> Any:
        return _drop_empty_values(value)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("заголовок пустой")
        return value

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
    def _field_not_in_both_lists(self) -> ProductCard:
        present = {_norm_key(k) for k in self.characteristics}
        both = [m for m in self.missing_fields if _norm_key(m) in present]
        if both:
            raise ValueError(
                "поле не может быть одновременно в characteristics и в missing_fields: "
                + ", ".join(both)
            )
        return self


class SupplierFacts(BaseModel):
    """Результат извлекателя: структурированные факты из текста или контекста."""

    model_config = ConfigDict(extra="ignore")

    product_name: str = ""
    characteristics: dict[str, str] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)

    @field_validator("characteristics", mode="before")
    @classmethod
    def _drop_empty(cls, value: Any) -> Any:
        return _drop_empty_values(value)


class CritiqueIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    field: str | None = None
    rule: str = ""
    message: str


class CritiqueReport(BaseModel):
    """Вердикт проверяющей роли. Literal превращается в enum в JSON-схеме."""

    model_config = ConfigDict(extra="ignore")

    verdict: Literal["approve", "regenerate"]
    issues: list[CritiqueIssue] = Field(default_factory=list)

    @field_validator("issues", mode="before")
    @classmethod
    def _issues_from_strings(cls, value: Any) -> Any:
        # Слабые модели возвращают замечания строками — принимаем и такую форму
        if isinstance(value, list):
            return [{"message": item} if isinstance(item, str) else item for item in value]
        return value


class InjectionVerdict(BaseModel):
    """Структурированный вердикт дешёвой модели-детектора инъекций."""

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


class CardRequest(BaseModel):
    supplier_text: str = Field(min_length=1)


class CardResponse(BaseModel):
    status: str
    attempts: int
    card: ProductCard | None = None
    issues: list[str] = Field(default_factory=list)
    review_reason: str | None = None


class GenerateCardRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1)
    product_hint: str | None = None
