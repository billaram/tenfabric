"""SQLite-backed run history tracking."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _default_db_path() -> Path:
    p = Path.home() / ".tenfabric" / "runs.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


class RunStore:
    """Lightweight SQLite store for training run metadata."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or _default_db_path()
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    project TEXT,
                    model TEXT,
                    provider TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    config_json TEXT,
                    error TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def create(
        self,
        run_id: str,
        project: str,
        model: str,
        provider: str,
        status: str = "pending",
        config_json: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO runs (id, project, model, provider, status, config_json, started_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, project, model, provider, status, config_json, now, now),
            )

    def update(
        self,
        run_id: str,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as conn:
            if status:
                conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))
                if status in ("completed", "failed", "cancelled"):
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute("UPDATE runs SET finished_at = ? WHERE id = ?", (now, run_id))
            if error:
                conn.execute("UPDATE runs SET error = ? WHERE id = ?", (error, run_id))

    def get(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            return dict(row) if row else None

    def list_recent(self, limit: int = 10) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(row) for row in rows]
