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
    # stage1
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
        f"CREATE INDEX IF NOT EXISTS {DB.REM_STAGE_1.value}_pred ON {DB.REM_STAGE_1.value} ({DB.PRED_LABEL.value});"
    )

    # stage2
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {DB.REM_STAGE_2.value} (
            {DB.ID.value} TEXT PRIMARY KEY,
            {DB.GOLD_LABEL.value} INTEGER NOT NULL,
            {DB.VERDICT.value} TEXT NOT NULL,
            {DB.FINAL_LABEL.value} INTEGER NOT NULL,
            {DB.SUPPORT_EVIDENCE.value} TEXT,
            {DB.ERROR_EVIDENCE.value} TEXT,
            {DB.MISSING_EVIDENCE.value} TEXT,
            {DB.MEMORY.value} TEXT,
            {DB.RAW_JSON.value} TEXT,
            {DB.CREATED_AT.value} TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {DB.REM_STAGE_2.value}_verdict ON {DB.REM_STAGE_2.value} ({DB.VERDICT.value});"
    )
    conn.commit()


def exists(conn: sqlite3.Connection, table: str, sid: str) -> bool:
    cur = conn.execute(
        f"SELECT 1 FROM {table} WHERE {DB.ID.value}=? LIMIT 1",
        (sid,),
    )
    return cur.fetchone() is not None


def upsert_stage1(
    conn: sqlite3.Connection,
    sid: str,
    pred_label: int,
    confidence: float,
    rationale: str,
    raw_json: str,
):
    conn.execute(
        f"""
        INSERT OR REPLACE INTO {DB.REM_STAGE_1.value}
        ({DB.ID.value}, {DB.PRED_LABEL.value}, {DB.CONFIDENCE.value}, {DB.RATIONALE.value}, {DB.RAW_JSON.value}, {DB.CREATED_AT.value})
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


def get_stage1(conn: sqlite3.Connection, sid: str):
    cur = conn.execute(
        f"""
        SELECT {DB.PRED_LABEL.value}, {DB.CONFIDENCE.value}, {DB.RATIONALE.value}
        FROM {DB.REM_STAGE_1.value}
        WHERE {DB.ID.value}=?
        """,
        (sid,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "pred_label": int(row[0]),
        "confidence": float(row[1]),
        "rationale": str(row[2]),
    }


def upsert_stage2(
    conn: sqlite3.Connection,
    *,
    sid: str,
    gold_label: int,
    verdict: str,
    final_label: int,
    support_evidence: str,
    error_evidence: str,
    missing_evidence: str,
    memory: str,
    raw_json: str,
):
    conn.execute(
        f"""
        INSERT OR REPLACE INTO {DB.REM_STAGE_2.value}
        (
            {DB.ID.value},
            {DB.GOLD_LABEL.value},
            {DB.VERDICT.value},
            {DB.FINAL_LABEL.value},
            {DB.SUPPORT_EVIDENCE.value},
            {DB.ERROR_EVIDENCE.value},
            {DB.MISSING_EVIDENCE.value},
            {DB.MEMORY.value},
            {DB.RAW_JSON.value},
            {DB.CREATED_AT.value}
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sid,
            int(gold_label),
            str(verdict),
            int(final_label),
            str(support_evidence),
            str(error_evidence),
            str(missing_evidence),
            str(memory),
            raw_json,
            datetime.utcnow().isoformat(),
        ),
    )
