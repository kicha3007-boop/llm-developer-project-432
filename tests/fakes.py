"""Подделки для тестов: эмбеддер без модели и помощники для живой базы."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from app.core.config import settings
from app.core.db import connection

DIM = settings.embed_dim


def unit(*dims: int, weights: Sequence[float] | None = None) -> list[float]:
    """Заданный руками вектор: единицы на указанных координатах, нормированный."""
    vec = [0.0] * DIM
    for i, d in enumerate(dims):
        vec[d] = weights[i] if weights else 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class HashEmbedder:
    """Мешок слов в хэш-пространстве: детерминирован, без загрузки модели."""

    model_name = "test-hash-embedder"
    dim = DIM

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for tok in re.findall(r"\w+", text.lower()):
            vec[int(hashlib.md5(tok.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def delete_document(document_id: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))


def delete_job(job_id: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM jobs WHERE id = %s", (job_id,))
        conn.execute("DELETE FROM llm_calls WHERE job_id = %s", (job_id,))
