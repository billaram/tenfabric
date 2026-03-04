"""Tests for SQLite run store."""

from __future__ import annotations

from pathlib import Path

import pytest

from tenfabric.core.run_store import RunStore


@pytest.fixture
def store(tmp_path: Path) -> RunStore:
    return RunStore(db_path=tmp_path / "test_runs.db")


def test_create_and_get(store: RunStore):
    store.create("run-001", project="test", model="llama-1b", provider="local")
    run = store.get("run-001")
    assert run is not None
    assert run["project"] == "test"
    assert run["status"] == "pending"


def test_update_status(store: RunStore):
    store.create("run-002", project="test", model="llama-1b", provider="local")
    store.update("run-002", status="completed")
    run = store.get("run-002")
    assert run["status"] == "completed"
    assert run["finished_at"] is not None


def test_update_error(store: RunStore):
    store.create("run-003", project="test", model="llama-1b", provider="local")
    store.update("run-003", status="failed", error="OOM")
    run = store.get("run-003")
    assert run["status"] == "failed"
    assert run["error"] == "OOM"


def test_list_recent(store: RunStore):
    for i in range(5):
        store.create(f"run-{i:03d}", project="test", model="llama", provider="local")
    runs = store.list_recent(limit=3)
    assert len(runs) == 3


def test_get_nonexistent(store: RunStore):
    assert store.get("nonexistent") is None
