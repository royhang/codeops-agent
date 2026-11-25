from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from codeops_agent.agent.state import AgentState


class SQLiteCheckpointStore:
    """Persists and restores AgentState snapshots by task_id for resumable execution."""

    def __init__(self, db_path: Path | str = ".codeops_state.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_checkpoints (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save(self, state: AgentState) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_checkpoints (task_id, status, state_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status = excluded.status,
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (state.task_id, state.status, state.model_dump_json(), now),
            )
            conn.commit()

    def load(self, task_id: str) -> Optional[AgentState]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM agent_checkpoints WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return AgentState.model_validate_json(row[0])

    def list_tasks(self, limit: int = 20) -> List[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT task_id, status, updated_at FROM agent_checkpoints ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"task_id": r[0], "status": r[1], "updated_at": r[2]} for r in rows]
