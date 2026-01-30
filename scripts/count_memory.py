# scripts/print_debate_memory_counts.py

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Optional

from config.config import Config, ModelEnum


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


def build_model_name_candidates() -> list[str]:
    names = [m.name for m in ModelEnum]
    names.sort(key=len, reverse=True)
    return names


MODEL_NAME_CANDIDATES = build_model_name_candidates()


def parse_dataset_and_model(stem: str) -> Tuple[str, str]:
    """
    파일명 규칙: {dataset}_{ModelEnum.name}.jsonl
    ModelEnum.name에 '_'가 포함될 수 있으므로,
    stem이 '_{model_name}'로 끝나는지 후보를 순회하며 매칭합니다.
    """
    for model_name in MODEL_NAME_CANDIDATES:
        suffix = "_" + model_name
        if stem.endswith(suffix):
            ds = stem[: -len(suffix)]
            if ds:
                return ds, model_name
    if "_" not in stem:
        return stem, "(unknown)"
    ds, model = stem.rsplit("_", 1)
    return ds, model


def count_memory_jsonl(path: Path) -> Tuple[int, int, int, int]:
    """
    returns:
      - n_lines_valid_json
      - n_unique_ids
      - n_bad_or_missing_id
      - n_missing_agents (agent_A/B/C/D 중 하나라도 없는 레코드 수)
    """
    valid = 0
    bad_id = 0
    missing_agents = 0
    seen = set()

    for rec in safe_iter_jsonl(path):
        valid += 1

        sid = rec.get("id")
        if sid is None:
            bad_id += 1
        else:
            seen.add(str(sid))

        if not (
            isinstance(rec, dict)
            and "agent_A" in rec
            and "agent_B" in rec
            and "agent_C" in rec
            and "agent_D" in rec
        ):
            missing_agents += 1

    return valid, len(seen), bad_id, missing_agents


def main():
    config = Config()
    memory_root = Path(config.memory_dir)

    if not memory_root.is_dir():
        print(f"[ERROR] memory_dir not found: {memory_root}")
        return

    files = sorted(memory_root.glob("*.jsonl"))
    if not files:
        print(f"[INFO] no jsonl files in: {memory_root}")
        return

    rows = []
    by_dataset: Dict[str, int] = defaultdict(int)

    for p in files:
        ds, model = parse_dataset_and_model(p.stem)
        n_valid, n_unique, n_bad_id, n_missing_agents = count_memory_jsonl(p)
        rows.append((ds, model, n_unique, n_valid, n_bad_id, n_missing_agents, p.name))
        by_dataset[ds] += n_unique

    rows.sort(key=lambda x: (x[0], x[1]))

    print(f"[memory_root] {memory_root}")
    print("-" * 130)
    print(
        f"{'DATASET':24s} {'MODEL':22s} {'UNIQUE_IDS':10s} {'VALID_LINES':10s} {'BAD_ID':6s} {'MISS_ABCD':9s} FILE"
    )
    print("-" * 130)
    for ds, model, n_unique, n_valid, n_bad_id, n_missing_agents, fname in rows:
        print(
            f"{ds:24s} {model:22s} {n_unique:10d} {n_valid:10d} {n_bad_id:6d} {n_missing_agents:9d} {fname}"
        )

    print("-" * 130)
    print("[dataset totals] (sum of unique ids across model files)")
    for ds in sorted(by_dataset.keys()):
        print(f"  - {ds}: {by_dataset[ds]}")
    print("-" * 130)


if __name__ == "__main__":
    main()
