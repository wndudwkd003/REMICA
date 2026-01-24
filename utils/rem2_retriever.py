# utils/rem2_retriever.py (핵심 부분만, 위쪽 Rem2Retriever 등은 그대로 두시고
# 아래 클래스 정의만 교체하시면 됩니다.)

from __future__ import annotations

from bisect import bisect_right
from typing import Any, Dict, List, Optional, Union

from torch.utils.data import Dataset

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
        self.conn = open_db(self.db_path)

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
        *,
        query_sid: str,
        query_text: str,
        top_k: int = 3,
        use_rem2: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        [
          {
            "sid": ...,
            "text": ...,
            "true_label": 0/1,
            "stage1_pred": 0/1,
            "stage1_rationale": "...",
            "stage2_is_correct": 0/1 or None,
            "stage2_evidence": "..." or "",
          },
          ...
        ]
        """
        # 1) 텍스트 유사도 기반 전체 검색 (label 구분 없음)
        sim_items = self.sim_retriever.get_similar_texts(
            ds=ds,
            query_sid=query_sid,
            query_text=query_text,
            top_k=top_k,
        )

        examples: List[Dict[str, Any]] = []

        for ex in sim_items:
            sid_ex = ex["sid"]
            text_ex = ex["text"]
            label_ex = int(ex["label"])  # train 라벨

            # REM1 (stage1)
            st1 = get_stage1(self.conn, sid_ex)
            if st1 is None:
                continue

            stage1_pred = int(st1["pred_label"])
            stage1_rationale = st1["rationale"] or ""

            # REM2 (stage2) 옵션
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

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


class Rem2ExampleAugDataset(Dataset):
    def __init__(
        self,
        datasets,
        retriever: Rem2Retriever,
        top_k=3,
        mode="rem1",  # "simple" | "rem1" | "rem12"
    ):
        assert mode in ("simple", "rem1", "rem12")
        self.datasets = datasets
        self.retriever = retriever
        self.top_k = int(top_k)
        self.mode = mode

        # global idx → (dataset_idx, local_idx) 매핑용 누적 길이
        self.cum_sizes = []
        total = 0
        for _, ds in self.datasets:
            total += len(ds)
            self.cum_sizes.append(total)

    def __len__(self):
        return self.cum_sizes[-1] if self.cum_sizes else 0

    def _locate(self, idx):
        ds_idx = bisect_right(self.cum_sizes, idx)
        prev_total = 0 if ds_idx == 0 else self.cum_sizes[ds_idx - 1]
        local_idx = idx - prev_total
        ds_enum, ds_obj = self.datasets[ds_idx]
        return ds_enum, ds_obj, local_idx

    def __getitem__(self, idx):
        ds_enum, base, local_idx = self._locate(idx)
        sid, text, label, meta = base[local_idx]

        examples = self.retriever.get_examples(
            ds=ds_enum,
            query_sid=sid,
            query_text=text,
            top_k=self.top_k,
            use_rem2=(self.mode == "rem12"),
        )

        target_block = f"TARGET_TEXT:\n{text}\n"

        if examples:
            blocks = []
            for i, ex in enumerate(examples, start=1):
                # 공통: 라벨 문자열
                t_lbl_word = "appropriate" if ex["true_label"] == 0 else "inappropriate"

                if self.mode == "simple":
                    # ---- rem_mode = "simple": 아주 단순한 포맷 ----
                    lines = [
                        f"EXAMPLE {i}:",
                        f"TEXT: {ex['text']}",
                        f"LABEL: {t_lbl_word}",  # 또는 숫자 label 쓰고 싶으면 ex['true_label']
                    ]

                else:
                    # rem1 / rem12 공통 부분
                    s1_lbl_word = "appropriate" if ex["stage1_pred"] == 0 else "inappropriate"

                    lines = [
                        f"EXAMPLE {i}:",
                        f"- ORIGINAL_TEXT: {ex['text']}",
                        f"- TRUE_LABEL: {t_lbl_word}",
                        f"- REM_STAGE1_PRED_LABEL: {s1_lbl_word}",
                        f"- REM_STAGE1_RATIONALE: {ex['stage1_rationale']}",
                    ]

                    if self.mode == "rem12":
                        lines.append(f"- REM_STAGE2_EVIDENCE: {ex['stage2_evidence']}")

                blocks.append("\n".join(lines))

            ex_block = "\n\n".join(blocks)
            full_text = target_block + "\n\n" + ex_block
        else:
            full_text = target_block

        return sid, full_text, label, meta
