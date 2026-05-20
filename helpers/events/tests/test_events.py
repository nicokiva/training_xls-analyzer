"""Tests for the event consumer (routine-analyzer side)."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build an in-memory (temp file) DB for each test
# ---------------------------------------------------------------------------

def _make_temp_db():
    """Create a temp SQLite file with the events table and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    conn = sqlite3.connect(tmp.name)
    conn.execute("""
        CREATE TABLE events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type   TEXT NOT NULL,
            payload      TEXT,
            created_at   TEXT NOT NULL,
            processed_at TEXT
        )
    """)
    conn.commit()
    conn.close()
    return Path(tmp.name)


def _insert_event(db_path, event_type, processed_at=None):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO events (event_type, created_at, processed_at) VALUES (?, '2026-01-01T00:00:00+00:00', ?)",
        (event_type, processed_at)
    )
    conn.commit()
    conn.close()


def _fetch_all(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM events").fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_consume_pending_returns_only_pending():
    db = _make_temp_db()
    _insert_event(db, "run:global")                          # pending
    _insert_event(db, "run:monthly", "2026-01-01T00:01:00") # already processed

    with patch("helpers.events.events.DB_PATH", db):
        from helpers.events.events import consume_pending_events
        events = consume_pending_events()

    assert len(events) == 1
    assert events[0]["event_type"] == "run:global"


def test_consume_pending_empty_when_all_processed():
    db = _make_temp_db()
    _insert_event(db, "run:global", "2026-01-01T00:01:00")

    with patch("helpers.events.events.DB_PATH", db):
        from helpers.events.events import consume_pending_events
        events = consume_pending_events()

    assert events == []


def test_consume_pending_returns_in_order():
    db = _make_temp_db()
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO events (event_type, created_at) VALUES ('run:global',  '2026-01-01T00:00:01+00:00')")
    conn.execute("INSERT INTO events (event_type, created_at) VALUES ('run:monthly', '2026-01-01T00:00:02+00:00')")
    conn.commit()
    conn.close()

    with patch("helpers.events.events.DB_PATH", db):
        from helpers.events.events import consume_pending_events
        events = consume_pending_events()

    assert [e["event_type"] for e in events] == ["run:global", "run:monthly"]


def test_mark_event_processed_sets_timestamp():
    db = _make_temp_db()
    _insert_event(db, "run:new-routine")

    with patch("helpers.events.events.DB_PATH", db):
        from helpers.events.events import consume_pending_events, mark_event_processed
        events = consume_pending_events()
        mark_event_processed(events[0]["id"])

        remaining = consume_pending_events()

    assert remaining == []
    rows = _fetch_all(db)
    assert rows[0]["processed_at"] is not None


def test_consume_pending_empty_db():
    db = _make_temp_db()

    with patch("helpers.events.events.DB_PATH", db):
        from helpers.events.events import consume_pending_events
        events = consume_pending_events()

    assert events == []
