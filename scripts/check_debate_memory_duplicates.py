# scripts/check_debate_memory_duplicates.py

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from config.config import Config


def safe_iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    config = Config()
    path = Path(config.memory_dir) / "ToxiSpanSE_GPT5_1.jsonl"

    ids = []
    for rec in safe_iter_jsonl(path):
        if "id" in rec:
            ids.append(str(rec["id"]))

    c = Counter(ids)
    n_valid = len(ids)
    n_unique = len(c)
    n_dup_lines = n_valid - n_unique
    top_dups = [(k, v) for k, v in c.items() if v > 1]
    top_dups.sort(key=lambda x: x[1], reverse=True)

    print(f"path: {path}")
    print(f"valid_lines(with id): {n_valid}")
    print(f"unique_ids: {n_unique}")
    print(f"duplicate_lines: {n_dup_lines}")
    print(f"n_ids_with_dup: {len(top_dups)}")

    print("\nTop 20 duplicated IDs:")
    for k, v in top_dups[:20]:
        print(f"  - id={k} count={v}")


if __name__ == "__main__":
    main()
