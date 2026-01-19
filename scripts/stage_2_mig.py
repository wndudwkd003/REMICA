# scripts/reset_stage2_schema.py
from __future__ import annotations

import sqlite3
from datetime import datetime

DB_PATH = "/kjy/llm_rag/multi_vp/rem/remica.sqlite3"
T_STAGE2 = "rem_stage_2"


def reset_stage2(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")

        # stage2 존재 확인
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (T_STAGE2,),
        ).fetchone()
        if row is None:
            print(f"[INFO] {T_STAGE2} not found -> create new.")
            conn.execute("BEGIN;")
        else:
            # 비어있는지 확인 (중요)
            n = conn.execute(f"SELECT COUNT(1) FROM {T_STAGE2};").fetchone()[0]
            if n != 0:
                raise RuntimeError(f"{T_STAGE2} is NOT empty (n={n}). Refuse to DROP.")

            conn.execute("BEGIN;")
            conn.execute(f"DROP TABLE IF EXISTS {T_STAGE2};")

        # 새 스키마 생성
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {T_STAGE2} (
                id TEXT PRIMARY KEY,
                true_label INTEGER NOT NULL,
                pred_label INTEGER NOT NULL,
                is_correct INTEGER NOT NULL,
                evidence TEXT,
                memory TEXT,
                raw_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

        # 인덱스 (iscorrect)
        conn.execute("DROP INDEX IF EXISTS rem_stage_2_verdict;")
        conn.execute("DROP INDEX IF EXISTS rem_stage_2_iscorrect;")
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS rem_stage_2_iscorrect ON {T_STAGE2} (is_correct);"
        )

        conn.execute("COMMIT;")
        print(f"[OK] {T_STAGE2} reset to NEW schema: {db_path}")

    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    reset_stage2(DB_PATH)
