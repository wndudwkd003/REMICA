# utils/db_utils.py

import sqlite3
from pathlib import Path
from params.db_value import DB
from datetime import datetime


def open_db(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    return conn


def init_schema(conn: sqlite3.Connection):
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DB.REM_STAGE_1.value} (
            {DB.ID.value} TEXT PRIMARY KEY,
            {DB.PRED_LABEL.value} INTEGER NOT NULL,
            {DB.CONFIDENCE.value} REAL,
            {DB.RATIONALE.value} TEXT,
            {DB.RAW_JSON.value} TEXT,
            {DB.CREATED_AT.value} TEXT NOT NULL
            )
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {DB.REM_STAGE_1.value}_pred
        ON {DB.REM_STAGE_1.value} ({DB.PRED_LABEL.value});
        """
    )

    conn.commit()


def exists(conn: sqlite3.Connection, sid: str):
    cur = conn.execute(
        f"SELECT 1 FROM {DB.REM_STAGE_1.value} WHERE {DB.ID.value}=? LIMIT 1", (sid,)
    )
    return cur.fetchone() is not None


def upsert(
    conn: sqlite3.Connection,
    sid: str,
    pred_label: int,
    confidence: float,
    rationale: str,
    raw_json: str,
):
    conn.execute(
        """
        INSERT OR REPLACE INTO rem_stage1
        (id, pred_label, confidence, rationale, raw_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            sid,
            int(pred_label),
            float(confidence),
            str(rationale),
            raw_json,
            datetime.utcnow().isoformat(),
        ),
    )
