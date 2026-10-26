"""Раннер SQL-миграций.

Миграции — обычные SQL-файлы в ``db/migrations`` вида ``0001_name.sql``. Раннер применяет их
по порядку, каждую в своей транзакции, и отмечает применённую версию в таблице
``schema_migrations`` (её создаёт первая миграция). Повторный запуск ничего не применяет.

Запуск: ``python -m app.core.migrate`` (или ``make migrate``).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg

from app.core.config import settings

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"
_NAME_RE = re.compile(r"^(\d{4})_[\w-]+\.sql$")


@dataclass(frozen=True)
class Migration:
    version: str
    name: str
    path: Path

    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


def discover(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """Список миграций каталога, отсортированный по версии. Дубли версий — ошибка."""
    found: dict[str, Migration] = {}
    for path in sorted(directory.glob("*.sql")):
        match = _NAME_RE.match(path.name)
        if not match:
            raise ValueError(f"Имя миграции не по шаблону NNNN_name.sql: {path.name}")
        version = match.group(1)
        if version in found:
            raise ValueError(f"Две миграции с версией {version}")
        found[version] = Migration(version=version, name=path.stem, path=path)
    return [found[v] for v in sorted(found)]


def pending(all_migrations: list[Migration], applied: set[str]) -> list[Migration]:
    return [m for m in all_migrations if m.version not in applied]


def applied_versions(conn: psycopg.Connection) -> set[str]:
    exists = conn.execute("SELECT to_regclass('public.schema_migrations') IS NOT NULL").fetchone()
    if not exists or not exists[0]:
        return set()
    return {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}


def migrate(database_url: str | None = None, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Применяет новые миграции и возвращает имена применённых.

    Раннер — отдельная короткоживущая команда, поэтому берёт одно своё соединение, а не пул
    приложения. Advisory-lock не даёт двум одновременным запускам применить одно и то же.
    """
    done: list[str] = []
    with psycopg.connect(database_url or settings.database_url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(727274)")
        try:
            todo = pending(discover(directory), applied_versions(conn))
            for migration in todo:
                with conn.transaction():
                    conn.execute(migration.sql())
                    conn.execute(
                        "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)",
                        (migration.version, migration.name),
                    )
                done.append(migration.name)
        finally:
            conn.execute("SELECT pg_advisory_unlock(727274)")
    return done


def main() -> int:
    applied = migrate()
    if applied:
        for name in applied:
            print(f"Применена миграция: {name}")
        print(f"Готово, применено миграций: {len(applied)}")
    else:
        print("Новых миграций нет")
    return 0


if __name__ == "__main__":
    sys.exit(main())
