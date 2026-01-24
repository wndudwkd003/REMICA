# utils/rem2_lm_dataset.py

from typing import Any, Dict

from torch.utils.data import Dataset

from config.config import Config
from params.db_value import DB
from utils.db_utils import get_stage1, open_db


def build_reasoning_target(
    label: int,
    reason_a: str,
    reason_b: str,
    is_correct: int | None,
) -> str:
    """
    label: 0/1
    reason_a: REM1 rationale
    reason_b: REM2 evidence
    is_correct: REM2에서 기록된 정오 정보 (1=맞음 → reinforcing, 0/None=틀림/미존재 → reflective)
    """
    reason_a = (reason_a or "").strip()
    reason_b = (reason_b or "").strip()

    if is_correct == 1:
        factor_phrase = "a reinforcing factor for my decision"
    else:
        factor_phrase = "a reflective factor that questions my initial decision"

    return (
        "<reasoning>"
        f"I initially think the correct answer is {label} because {reason_a}. "
        "Hmm, wait a minute. "
        f"After reconsidering the case and looking at the evidence, I realize that {reason_b} can act as {factor_phrase}. "
        f"In conclusion, the correct answer is {label}."
        "</reasoning>"
        f"<answer>{label}</answer>"
    )


def build_system_prompt(llm_target_mode: str) -> str:
    """
    llm_target_mode 에 따라 SYSTEM 프롬프트를 다르게 생성.
      - "simple": 간단한 reasoning + 정답만
      - 그 외: 기존 REM1/REM2 반영한 두 단계 추론 설명
    """
    if llm_target_mode == "simple":
        return """You are a rater who judges whether a text is appropriate or inappropriate.

Instructions:
- Read the TARGET_TEXT.
- Read the SIMILAR EXAMPLES as reference only (do not copy them).
- Decide one final label:
  - 0 = appropriate
  - 1 = inappropriate

Output format:
- Optionally write a short reasoning in English inside <reasoning>...</reasoning>.
- Then output the final label inside <answer>...</answer>.
- Keep the format strictly as:
  <reasoning>...your reasoning (can be short)...</reasoning><answer>0 or 1</answer>
"""

    # 기본/기존 모드 (rem12)
    return """You are a rater who judges whether a text is appropriate or inappropriate.

Instructions:
- Read the TARGET_TEXT.
- Read the SIMILAR EXAMPLES as reference only.
- Think in two phases: an initial guess, then a short "Hmm, wait a minute." reconsideration.
- After reflecting, decide one final label:
  - 0 = appropriate
  - 1 = inappropriate

Output format:
- Write your full reasoning in English inside <reasoning>...</reasoning>.
- Then output the final label inside <answer>...</answer>.
- Keep the format strictly as:
  <reasoning>...your detailed two-phase reasoning...</reasoning><answer>0 or 1</answer>
"""


class Rem2LMDataset(Dataset):
    """
    base_dataset: Rem2ExampleAugDataset 처럼
      __getitem__에서 (sid, full_text, label, meta)를 반환하는 객체.
      full_text 안에는 TEXT + SIMILAR TEXTS (reference only)가 이미 들어 있음.
    """

    def __init__(self, base_dataset, config: Config):
        self.base = base_dataset
        self.config = config
        self.db_path = config.remica_db_path

        # simple 모드에서는 DB를 안 써도 되므로 안 열어도 됨
        self.use_db = (self.config.llm_target_mode != "simple")

        if self.use_db:
            self.conn = open_db(self.db_path)
        else:
            self.conn = None

        # 모드에 맞는 SYSTEM 프롬프트를 미리 만들어 둠
        self.system_prompt = build_system_prompt(self.config.llm_target_mode)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, full_text, label, meta = self.base[idx]
        label_int = int(label)

        # -------------------------------
        # 1) target 생성: 모드에 따라 분기
        # -------------------------------
        if self.config.llm_target_mode == "simple":
            # 아주 단순한 타깃: 비워둔 reasoning + 최종 정답만
            target = f"<reasoning></reasoning><answer>{label_int}</answer>"

        else:
            # 기존 REM1/REM2 기반 reasoning 타깃
            reason_a = ""
            reason_b = ""
            is_correct = None

            if self.conn is not None:
                # stage1 정보 (REM1)
                st1 = get_stage1(self.conn, sid)
                if st1 is not None:
                    reason_a = st1.get("rationale", "") or ""

                # stage2 정보 (REM2)
                cur = self.conn.execute(
                    f"""
                    SELECT {DB.IS_CORRECT.value}, {DB.EVIDENCE.value}
                    FROM {DB.REM_STAGE_2.value}
                    WHERE {DB.ID.value}=?
                    LIMIT 1
                    """,
                    (sid,),
                )
                row = cur.fetchone()
                if row is not None:
                    is_correct = int(row[0]) if row[0] is not None else None
                    reason_b = row[1] or ""

            target = build_reasoning_target(
                label=label_int,
                reason_a=reason_a,
                reason_b=reason_b,
                is_correct=is_correct,
            )

        # -------------------------------
        # 2) 프롬프트 (SYSTEM + full_text)
        # -------------------------------
        prompt = f"{self.system_prompt}\n\n{full_text}"

        return {
            "sid": sid,
            "prompt": prompt,
            "target": target,
            "label": label_int,
            "meta": meta,
        }

    def close(self):
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass
