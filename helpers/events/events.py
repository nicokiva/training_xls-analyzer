"""
Event consumer for routine-analyzer.

SQLite is a file-based relational database — no server process, no installation.
The entire database lives in a single .db file on disk. It is ideal here because
both projects run on the same machine and only need a lightweight way to share
data without any network or daemon overhead.

The two projects (pdf2xls-generator and routine-analyzer) are decoupled via a
shared SQLite database that acts as a pub/sub event queue:

  - pdf2xls-generator publishes an event (e.g. "run:new-routine") to events.db.
  - routine-analyzer reads pending events on startup and processes each one,
    then marks them as processed so they are never executed twice.

This way neither project calls the other directly — they only share the DB file.

Event types published by pdf2xls-generator:
  "run:global"      — run the global analysis
  "run:monthly"     — run the monthly analysis
  "run:new-routine" — run the new-routine analysis
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from training_shared.events import EventType  # shared constants — no magic strings

DB_PATH = Path(__file__).parent.parent.parent.parent / "events.db"


def _get_connection():
    """Open (or create) the SQLite database and ensure the events table exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name, e.g. row["event_type"]

    # Create the table only if it doesn't already exist (safe to call every time).
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type   TEXT    NOT NULL,
            payload      TEXT,               -- optional JSON string for extra data
            created_at   TEXT    NOT NULL,
            processed_at TEXT                -- NULL means "pending"
        )
    """)
    conn.commit()
    return conn


def consume_pending_events():
    """
    Return all events that have not been processed yet, ordered by creation time.

    Each event is a sqlite3.Row, so you can do: event["event_type"], event["id"], etc.
    The caller is responsible for marking each event processed via mark_event_processed().
    """
    conn = _get_connection()
    # Select only rows where processed_at is still NULL (pending).
    rows = conn.execute(
        "SELECT * FROM events WHERE processed_at IS NULL ORDER BY created_at ASC"
    ).fetchall()
    conn.close()
    return rows


def mark_event_processed(event_id):
    """
    Stamp processed_at on the given event so it is never picked up again.

    Args:
        event_id: The integer id of the event row.
    """
    conn = _get_connection()
    now  = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE events SET processed_at = ? WHERE id = ?",
        (now, event_id)
    )
    conn.commit()
    conn.close()
