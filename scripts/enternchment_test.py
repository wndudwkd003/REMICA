from pathlib import Path
import json
import difflib
import re
import statistics

import numpy as np
from sentence_transformers import SentenceTransformer

from config.config import Config, TARGET_DATASETS
from utils.seeds_utils import set_seeds


MODELS_TO_EVAL = [
    "GPT5_1",
    "GPT5_MINI",
    "CLAUDE_HAIKU4_5",
]

CHAINS_TO_EVAL = [
    "chain_of_expert",
    "chain_of_debate",
]

# 임베딩 모델 이름 (원하시면 Config에서 가져오도록 바꿔도 됨)
RAG_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(RAG_MODEL_NAME)


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            items.append(json.loads(s))
    return items


def index_by_id(items):
    out = {}
    for obj in items:
        out[obj["id"]] = obj
    return out


def get_output(obj, key: str):
    return obj[key]["output"]


def get_rationale(obj, key: str):
    return str(get_output(obj, key)["rationale"])


def normalize_text(s: str):
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def similarity_seq(a: str, b: str):
    a2 = normalize_text(a)
    b2 = normalize_text(b)
    return difflib.SequenceMatcher(None, a2, b2).ratio()


def similarity_jac(a: str, b: str):
    a2 = normalize_text(a)
    b2 = normalize_text(b)
    wa = set(re.findall(r"[a-z0-9]+", a2))
    wb = set(re.findall(r"[a-z0-9]+", b2))
    if not wa and not wb:
        return 1.0
    inter = len(wa & wb)
    union = len(wa | wb)
    return inter / union


def similarity_emb(a: str, b: str):
    a2 = normalize_text(a)
    b2 = normalize_text(b)

    embeddings = embedding_model.encode(
        [a2, b2],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    v1 = embeddings[0]
    v2 = embeddings[1]

    # normalize_embeddings=True 이므로 내적이 곧 코사인 유사도
    return float(np.dot(v1, v2))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def has_rounds(obj, agent_prefix: str):
    k1 = f"{agent_prefix}1"
    k2 = f"{agent_prefix}2"
    return (k1 in obj) and (k2 in obj)


def round_delta(obj, agent_prefix: str, metric: str):
    r1 = get_rationale(obj, f"{agent_prefix}1")
    r2 = get_rationale(obj, f"{agent_prefix}2")

    if metric == "seq":
        sim = similarity_seq(r1, r2)
    elif metric == "jac":
        sim = similarity_jac(r1, r2)
    elif metric == "emb":
        sim = similarity_emb(r1, r2)
    else:
        raise ValueError(f"unknown metric: {metric}")

    return 1.0 - sim


def summarize_median(deltas):
    n = len(deltas)
    if n == 0:
        return {"n": 0, "median": 0.0}
    return {"n": n, "median": float(statistics.median(deltas))}


def collect_condition_deltas_for_dataset(
    model_dir_name: str,
    chain_name: str,
    dataset_name: str,
    agent_prefix: str,
    metric: str,
):
    memory_root = Path("debate_memory") / model_dir_name / chain_name
    off_path = memory_root / "ali_off" / f"{dataset_name}.jsonl"
    on_path = memory_root / "ali_on" / f"{dataset_name}.jsonl"

    ali_off_items = load_jsonl(off_path)
    ali_on_items = load_jsonl(on_path)

    off_by_id = index_by_id(ali_off_items)
    on_by_id = index_by_id(ali_on_items)

    off_deltas = []
    on_deltas = []

    for sid in off_by_id:
        if sid not in on_by_id:
            continue

        off_obj = off_by_id[sid]
        on_obj = on_by_id[sid]

        if not has_rounds(off_obj, agent_prefix):
            continue
        if not has_rounds(on_obj, agent_prefix):
            continue

        off_deltas.append(round_delta(off_obj, agent_prefix, metric))
        on_deltas.append(round_delta(on_obj, agent_prefix, metric))

    return off_deltas, on_deltas


def evaluate_model_chain(model_dir_name: str, chain_name: str):
    dataset_names = [ds.name for ds in TARGET_DATASETS]

    out = {
        "agent_A": {
            "seq": {"off": None, "on": None},
            "jac": {"off": None, "on": None},
            "emb": {"off": None, "on": None},
        },
        "agent_B": {
            "seq": {"off": None, "on": None},
            "jac": {"off": None, "on": None},
            "emb": {"off": None, "on": None},
        },
    }

    for agent_prefix, agent_key in [("A", "agent_A"), ("B", "agent_B")]:
        for metric in ["seq", "jac", "emb"]:
            off_all = []
            on_all = []

            for dataset_name in dataset_names:
                off_deltas, on_deltas = collect_condition_deltas_for_dataset(
                    model_dir_name=model_dir_name,
                    chain_name=chain_name,
                    dataset_name=dataset_name,
                    agent_prefix=agent_prefix,
                    metric=metric,
                )
                off_all.extend(off_deltas)
                on_all.extend(on_deltas)

            out[agent_key][metric]["off"] = summarize_median(off_all)
            out[agent_key][metric]["on"] = summarize_median(on_all)

    return out


def main():
    config = Config()

    results = {}

    for model_dir_name in MODELS_TO_EVAL:
        results[model_dir_name] = {}
        for chain_name in CHAINS_TO_EVAL:
            results[model_dir_name][chain_name] = evaluate_model_chain(
                model_dir_name=model_dir_name,
                chain_name=chain_name,
            )

    summary = {
        "results": results
    }

    out_dir = Path(config.runs_analysis_dir) / "enternchment"
    ensure_dir(out_dir)
    out_path = out_dir / "summary.json"
    save_json(out_path, summary)
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    config = Config()
    set_seeds(config.seed)
    main()
