# utils/rem2_retriever.py

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Optional, Union

from config.config import Config, DatasetEnum
from params.db_value import DB
from utils.db_utils import get_stage1, open_db
from utils.retriever import SimilarTextRetriever


class Rem2Retriever:
    def __init__(self, config: Config, device: str):
        self.config = config
        self.device = device

        # 유사 텍스트 검색용 (FAISS + SentenceTransformer)
        self.sim_retriever = SimilarTextRetriever(config, device=device)

        # REM1/REM2 DB
        self.db_path = config.remica_db_path
        self.conn: sqlite3.Connection = open_db(self.db_path)

    def get_stage2_one(self, sid: str) -> Optional[Dict[str, Any]]:
        table = DB.REM_STAGE_2.value
        cur = self.conn.execute(
            f"""
            SELECT {DB.IS_CORRECT.value}, {DB.EVIDENCE.value}
            FROM {table}
            WHERE {DB.ID.value}=?
            LIMIT 1
            """,
            (sid,),
        )
        row = cur.fetchone()
        if row is None:
            return None

        is_correct = int(row[0]) if row[0] is not None else 0
        evidence = row[1] or ""
        return {
            "sid": sid,
            "is_correct": is_correct,
            "evidence": evidence,
        }

    def get_examples(
        self,
        ds: Union[DatasetEnum, str],
        query_sid: str,
        query_text: str,
        top_k: int = 3,
        use_rem2: bool = False,
    ):

        sim_items = self.sim_retriever.get_similar_texts(
            ds=ds,
            query_sid=query_sid,
            query_text=query_text,
            top_k=top_k,
        )

        examples = []

        for ex in sim_items:
            sid_ex = ex["sid"]
            text_ex = ex["text"]
            label_ex = int(ex["label"])  # 실제 train 라벨

            # 2) REM1 (stage1) 조회
            st1 = get_stage1(self.conn, sid_ex)
            if st1 is None:
                # REM1이 없는 샘플은 스킵
                continue

            stage1_pred = int(st1["pred_label"])
            stage1_rationale = st1["rationale"] or ""

            # 3) REM2 (stage2) 조회 (옵션)
            stage2_is_correct: Optional[int] = None
            stage2_evidence: str = ""
            if use_rem2:
                st2 = self.get_stage2_one(sid_ex)
                if st2 is not None:
                    stage2_is_correct = int(st2["is_correct"])
                    stage2_evidence = st2["evidence"] or ""

            examples.append(
                {
                    "sid": sid_ex,
                    "text": text_ex,
                    "true_label": label_ex,
                    "stage1_pred": stage1_pred,
                    "stage1_rationale": stage1_rationale,
                    "stage2_is_correct": stage2_is_correct,
                    "stage2_evidence": stage2_evidence,
                }
            )

        return examples

    # -----------------------------
    # cleanup
    # -----------------------------
    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


class Rem2ExampleAugDataset:
    def __init__(
        self,
        base,
        retriever: Rem2Retriever,
        ds: DatasetEnum,
        top_k: int = 3,
        mode: str = "rem1",
    ):
        self.base = base
        self.retriever = retriever
        self.ds = ds
        self.top_k = top_k
        self.mode = mode

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sid, text, label, meta = self.base[idx]

        examples = self.retriever.get_examples(
            ds=self.ds,
            query_sid=sid,
            query_text=text,
            top_k=self.top_k,
            use_rem2=(self.mode == "rem12"),
        )

        target_block = f"{text}\n"

        if examples:
            blocks = []
            for i, ex in enumerate(examples, start=1):
                t_lbl = "appropriate" if ex["true_label"] == 0 else "inappropriate"
                s1_lbl = "appropriate" if ex["stage1_pred"] == 0 else "inappropriate"

                lines = [
                    f"EXAMPLE {i}:",
                    f"- ORIGINAL_TEXT: {ex['text']}",
                    f"- TRUE_LABEL: {t_lbl}",
                    f"- REM_STAGE1_PRED_LABEL: {s1_lbl}",
                    f"- REM_STAGE1_RATIONALE: {ex['stage1_rationale']}",
                ]

                if self.mode == "rem12":
                    lines.append(f"- REM_STAGE2_EVIDENCE: {ex['stage2_evidence']}")

                blocks.append("\n".join(lines))

            ex_block = "\n".join(blocks)
            full_text = target_block + "\n\n" + ex_block
        else:
            full_text = target_block

        return sid, full_text, label, meta
