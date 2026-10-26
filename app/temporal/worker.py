"""Точка входа воркера процессов: ``python -m app.temporal.worker`` (или ``make worker``).

Регистрирует процессы и все шаги. Синхронным шагам (база, диск, разбор, эмбеддинги) выдаётся
пул потоков — без него воркер откажется их брать.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from temporalio.worker import Worker

from app.core.config import settings
from app.core.db import close_pool, open_pool
from app.core.logging import get_logger, setup_logging
from app.temporal.activities import ALL_ACTIVITIES
from app.temporal.client import get_client
from app.temporal.workflows import ALL_WORKFLOWS

log = get_logger(__name__)


async def main() -> None:
    setup_logging()
    open_pool(wait=True)
    client = await get_client()
    with ThreadPoolExecutor(max_workers=8) as executor:
        worker = Worker(
            client,
            task_queue=settings.temporal_task_queue,
            workflows=ALL_WORKFLOWS,
            activities=ALL_ACTIVITIES,
            activity_executor=executor,
            max_concurrent_activities=8,
        )
        log.info("worker_started", task_queue=settings.temporal_task_queue)
        try:
            await worker.run()
        finally:
            close_pool()


if __name__ == "__main__":
    asyncio.run(main())
