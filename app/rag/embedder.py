"""Локальные эмбеддинги через sentence-transformers.

Модели семейства e5 / EmbeddingGemma обучены с задачными префиксами: запрос и документ кодируются
по-разному. Без префиксов близость релевантных пар проседает, и порог отсечения выметает верные
результаты. Префиксы — часть контракта модели, они берутся из её карточки и живут в настройках.

Модель грузится один раз на процесс (лениво, при первом вызове), векторы считаются пачкой
и нормируются по L2 — тогда косинусная близость совпадает со скалярным произведением.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, Protocol

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def document_input(text: str, section: str | None = None, title: str | None = None) -> str:
    """Текст фрагмента для кодирования: секция и заголовок документа добавляют контекст."""
    head = " / ".join(x for x in (title, section) if x)
    return f"{head}. {text}" if head else text


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = settings.embedding_model,
        dim: int = settings.embedding_dim,
        query_prefix: str = settings.embedding_query_prefix,
        document_prefix: str = settings.embedding_document_prefix,
        batch_size: int = settings.embedding_batch_size,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    log.info("embedding_model_loading", model=self.model_name)
                    model = SentenceTransformer(self.model_name, device="cpu")
                    get_dim = (
                        getattr(model, "get_embedding_dimension", None)
                        or model.get_sentence_embedding_dimension
                    )
                    actual = get_dim()
                    if actual != self.dim:
                        raise RuntimeError(
                            f"Размерность модели {self.model_name} = {actual}, а в настройках и "
                            f"миграции — {self.dim}. Нужна новая миграция и переиндексация."
                        )
                    self._model = model
                    log.info("embedding_model_loaded", model=self.model_name, dim=actual)
        return self._model

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Эмбеддер процесса (один на процесс). Тесты подменяют его через ``set_embedder``."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
    return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    global _embedder
    _embedder = embedder
