# utils/db_utils.py
import sqlite3
from datetime import datetime

from params.db_value import DB


def open_db(path: str):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_stage1_schema(conn: sqlite3.Connection):
    table = DB.REM_STAGE_1.value
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            {DB.ID.value} TEXT PRIMARY KEY,
            {DB.PRED_LABEL.value} INTEGER NOT NULL,
            {DB.RATIONALE.value} TEXT,
            {DB.RAW_JSON.value} TEXT,
            {DB.CREATED_AT.value} TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_pred ON {table} ({DB.PRED_LABEL.value});"
    )
    conn.commit()


def _get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return {str(r[1]) for r in cur.fetchall()}  # r[1] = column name


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, col_def: str):
    cols = _get_columns(conn, table)
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_def};")


def init_stage2_schema(conn: sqlite3.Connection):
    table = DB.REM_STAGE_2.value

    # 1) 테이블이 없으면 최신 스키마로 생성
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            {DB.ID.value} TEXT PRIMARY KEY,
            {DB.IS_CORRECT.value} INTEGER NOT NULL,
            {DB.EVIDENCE.value} TEXT,
            {DB.RAW_JSON.value} TEXT,
            {DB.CREATED_AT.value} TEXT NOT NULL
        )
        """
    )

    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_iscorrect ON {table} ({DB.IS_CORRECT.value});"
    )
    conn.commit()


def init_ica_schema(conn: sqlite3.Connection):
    table = DB.ICA.value
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            {DB.ID.value} TEXT PRIMARY KEY,

            {DB.SOURCE_CID.value} TEXT NOT NULL,
            {DB.SOURCE_FILE.value} TEXT NOT NULL,
            {DB.WINDOW_START.value} INTEGER NOT NULL,
            {DB.WINDOW_END.value} INTEGER NOT NULL,
            {DB.LAST_TEXT.value} TEXT NOT NULL,

            {DB.CONVERSATION_LABEL.value} INTEGER NOT NULL,

            {DB.CONTEXT_SUMMARY.value} TEXT,
            {DB.TRIGGERS_JSON.value} TEXT,
            {DB.TARGETS_JSON.value} TEXT,
            {DB.RULES_JSON.value} TEXT,

            {DB.RAW_JSON.value} TEXT,
            {DB.CREATED_AT.value} TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_ds_split "
        f"ON {table} ({DB.SOURCE_DATASET.value}, {DB.SPLIT.value});"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_sourcecid "
        f"ON {table} ({DB.SOURCE_CID.value});"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_win "
        f"ON {table} ({DB.SOURCE_CID.value}, {DB.WINDOW_START.value});"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table}_clabel "
        f"ON {table} ({DB.CONVERSATION_LABEL.value});"
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
    rationale: str,
    raw_json: str,
):
    table = DB.REM_STAGE_1.value
    conn.execute(
        f"""
        INSERT OR REPLACE INTO {table}
        ({DB.ID.value}, {DB.PRED_LABEL.value}, {DB.RATIONALE.value}, {DB.RAW_JSON.value}, {DB.CREATED_AT.value})
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            sid,
            pred_label,
            rationale,
            raw_json,
            datetime.utcnow().isoformat(),
        ),
    )


def get_stage1(conn: sqlite3.Connection, sid: str):
    table = DB.REM_STAGE_1.value
    cur = conn.execute(
        f"""
        SELECT {DB.PRED_LABEL.value}, {DB.RATIONALE.value}
        FROM {table}
        WHERE {DB.ID.value}=?
        """,
        (sid,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {"pred_label": int(row[0]),"rationale": row[1]}


def upsert_stage2(
    conn: sqlite3.Connection,
    *,
    sid: str,
    is_correct: int,
    evidence: str,
    raw_json: str,
):
    table = DB.REM_STAGE_2.value
    conn.execute(
        f"""
        INSERT OR REPLACE INTO {table}
        ({DB.ID.value}, {DB.IS_CORRECT.value}, {DB.EVIDENCE.value}, {DB.RAW_JSON.value}, {DB.CREATED_AT.value})
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            sid,
            int(is_correct),
            evidence,
            raw_json,
            datetime.utcnow().isoformat(),
        ),
    )


def get_stage2(conn: sqlite3.Connection, sid: str):
    table = DB.REM_STAGE_2.value
    cur = conn.execute(
        f"""
        SELECT {DB.TRUE_LABEL.value}, {DB.PRED_LABEL.value}, {DB.IS_CORRECT.value},
               {DB.EVIDENCE.value}, {DB.MEMORY.value}, {DB.RELIABILITY.value},
               {DB.RUNS_JSON.value}, {DB.RAW_JSON.value}
        FROM {table}
        WHERE {DB.ID.value}=?
        """,
        (sid,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "true_label": int(row[0]),
        "pred_label": int(row[1]),
        "is_correct": int(row[2]),
        "evidence": row[3],
        "memory": row[4],
        "reliability": float(row[5]) if row[5] is not None else None,
        "runs_json": row[6],
        "raw_json": row[7],
    }


def upsert_ica(
    conn: sqlite3.Connection,
    *,
    sid: str,
    source_dataset: str,
    split: str,
    source_cid: str,
    source_file: str,
    window_start: int,
    window_end: int,
    last_text: str,
    conversation_label: int,
    context_summary: str,
    triggers_json: str,
    targets_json: str,
    rules_json: str,
    raw_json: str,
):
    table = DB.ICA.value
    conn.execute(
        f"""
        INSERT OR REPLACE INTO {table}
        ({DB.ID.value},
         {DB.SOURCE_DATASET.value}, {DB.SPLIT.value},

         {DB.SOURCE_CID.value}, {DB.SOURCE_FILE.value},
         {DB.WINDOW_START.value}, {DB.WINDOW_END.value}, {DB.LAST_TEXT.value},

         {DB.CONVERSATION_LABEL.value},

         {DB.CONTEXT_SUMMARY.value}, {DB.TRIGGERS_JSON.value}, {DB.TARGETS_JSON.value}, {DB.RULES_JSON.value},
         {DB.RAW_JSON.value}, {DB.CREATED_AT.value})
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sid,
            source_dataset,
            split,
            source_cid,
            source_file,
            int(window_start),
            int(window_end),
            last_text,
            int(conversation_label),
            context_summary,
            triggers_json,
            targets_json,
            rules_json,
            raw_json,
            datetime.utcnow().isoformat(),
        ),
    )


def get_ica(conn: sqlite3.Connection, sid: str):
    table = DB.ICA.value
    cur = conn.execute(
        f"""
        SELECT
            {DB.SOURCE_DATASET.value}, {DB.SPLIT.value},
            {DB.SOURCE_CID.value}, {DB.SOURCE_FILE.value},
            {DB.WINDOW_START.value}, {DB.WINDOW_END.value}, {DB.LAST_TEXT.value},
            {DB.CONVERSATION_LABEL.value},
            {DB.CONTEXT_SUMMARY.value}, {DB.TRIGGERS_JSON.value}, {DB.TARGETS_JSON.value}, {DB.RULES_JSON.value},
            {DB.RAW_JSON.value}
        FROM {table}
        WHERE {DB.ID.value}=?
        """,
        (sid,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "source_dataset": row[0],
        "split": row[1],
        "source_cid": row[2],
        "source_file": row[3],
        "window_start": int(row[4]),
        "window_end": int(row[5]),
        "last_text": row[6],
        "conversation_label": int(row[7]),
        "context_summary": row[8],
        "triggers_json": row[9],
        "targets_json": row[10],
        "rules_json": row[11],
        "raw_json": row[12],
    }
