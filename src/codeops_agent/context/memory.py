from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from pydantic import BaseModel


class MemoryFact(BaseModel):
    namespace: str
    fact_key: str
    content: str
    created_at: str


class DualLayerMemory:
    """Short-term sliding window + SQLite-backed long-term troubleshooting memory."""

    def __init__(self, db_path: Path | str = ".codeops_state.db", short_term_limit: int = 8) -> None:
        self.db_path = str(db_path)
        self.short_term_limit = short_term_limit
        self._short_term: List[str] = []
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    fact_key TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(namespace, fact_key)
                )
                """
            )
            conn.commit()

    def add_short_term(self, item: str) -> None:
        self._short_term.append(item)
        if len(self._short_term) > self.short_term_limit:
            self._short_term = self._short_term[-self.short_term_limit :]

    def get_short_term(self) -> List[str]:
        return list(self._short_term)

    def upsert_long_term(self, namespace: str, fact_key: str, content: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO long_term_memory (namespace, fact_key, content, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, fact_key) DO UPDATE SET
                    content = excluded.content,
                    created_at = excluded.created_at
                """,
                (namespace, fact_key, content, now),
            )
            conn.commit()

    def recall_long_term(self, namespace: str, query: str, limit: int = 3) -> List[MemoryFact]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT namespace, fact_key, content, created_at
                FROM long_term_memory
                WHERE namespace = ?
                ORDER BY id DESC
                """,
                (namespace,),
            ).fetchall()

        query_terms = [t.lower() for t in query.split() if len(t) > 1]
        scored: List[tuple[int, MemoryFact]] = []
        for ns, fk, content, created_at in rows:
            hay = f"{fk} {content}".lower()
            hits = sum(1 for term in query_terms if term in hay)
            scored.append((hits, MemoryFact(namespace=ns, fact_key=fk, content=content, created_at=created_at)))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [fact for _, fact in scored[:limit]]
