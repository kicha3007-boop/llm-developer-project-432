"""Пул соединений с PostgreSQL — ресурс уровня приложения.

Пул создаётся один раз на процесс (на старте приложения или воркера) и закрывается при остановке.
Операция берёт соединение из пула контекстным менеджером ``connection()`` и сразу возвращает его.
Соединение берут на операцию, а не на весь шаг процесса: шаг генерации ждёт модель минутами.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg_pool import ConnectionPool

from app.core.config import settings

_pool: ConnectionPool | None = None
_lock = threading.Lock()


def pool() -> ConnectionPool:
    """Отдаёт пул соединений процесса, создавая его при первом обращении."""
    global _pool
    if _pool is None:
        with _lock:
            if _pool is None:
                _pool = ConnectionPool(
                    conninfo=settings.database_url,
                    min_size=settings.db_pool_min_size,
                    max_size=settings.db_pool_max_size,
                    timeout=settings.db_pool_timeout_s,
                    # изменения фиксируются сразу: статус задачи виден клиенту немедленно
                    kwargs={"autocommit": True},
                    open=True,
                    name="card-pool",
                )
    return _pool


def pool_initialized() -> bool:
    """Создан ли пул в этом процессе (без побочного эффекта создания)."""
    return _pool is not None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Соединение из пула на одну операцию; по выходе из блока возвращается в пул."""
    with pool().connection() as conn:
        yield conn


def open_pool(wait: bool = False) -> ConnectionPool:
    """Создаёт пул на старте процесса. ``wait=True`` дожидается первого соединения."""
    p = pool()
    if wait:
        p.wait(timeout=settings.db_pool_timeout_s)
    return p


def close_pool() -> None:
    """Освобождает пул при остановке процесса."""
    global _pool
    with _lock:
        if _pool is not None:
            _pool.close(timeout=1.0)
            _pool = None


def readiness() -> dict:
    """Проверка готовности: база отвечает и расширение vector активно.

    Это инфраструктурный запрос уровня ядра (как ping), а не предметный SQL.
    """
    with connection() as conn:
        conn.execute("SELECT 1")
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    return {"db": "ok", "vector": row[0] if row else None}
