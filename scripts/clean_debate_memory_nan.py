# scripts/clean_debate_memory_nan.py
from __future__ import annotations

import json
import re
from pathlib import Path

IN_PATH = Path("debate_memory/ToxiSpanSE_GPT5_1.jsonl")
OUT_PATH = Path("debate_memory/ToxiSpanSE_GPT5_1.cleaned.jsonl")

NORMALIZE_FLOAT_SUFFIX = True  # "xxx_123.0" -> "xxx_123"
DEDUP_BY_ID = False  # 필요하면 True

float_suffix_re = re.compile(r"^(.*?)(\d+)\.0$")


def is_bad_id(x) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    if s == "":
        return True
    # "nan", "NaN", "...nan..." 전부 제거
    if "nan" in s.lower():
        return True
    return False


def normalize_id(s: str) -> str:
    s = s.strip()
    if not NORMALIZE_FLOAT_SUFFIX:
        return s
    m = float_suffix_re.match(s)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return s


def main():
    in_lines = IN_PATH.read_text(encoding="utf-8").splitlines()

    kept = 0
    bad_json = 0
    bad_id = 0
    dup = 0

    seen = set()
    out_f = OUT_PATH.open("w", encoding="utf-8")

    for line in in_lines:
        if not line.strip():
            continue

        try:
            obj = json.loads(line)
        except Exception:
            bad_json += 1
            continue

        _id = obj.get("id", None)
        if is_bad_id(_id):
            bad_id += 1
            continue

        new_id = normalize_id(str(_id))
        obj["id"] = new_id

        if DEDUP_BY_ID:
            if new_id in seen:
                dup += 1
                continue
            seen.add(new_id)

        out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        kept += 1

    out_f.close()

    print(
        f"[DONE] in={len(in_lines)} kept={kept} bad_id={bad_id} bad_json={bad_json} dup={dup}"
    )
    print(f"[OUT]  {OUT_PATH}")


if __name__ == "__main__":
    main()
