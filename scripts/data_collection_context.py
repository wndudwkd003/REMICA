# scripts/data_collection_context.py
from __future__ import annotations

import json
import html
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List
import random

import matplotlib.pyplot as plt


class DatasetEnum(Enum):
    PROSOCIAL_DIALOG = "prosocial-dialog"
    BOT_ADVERSARIAL_DIALOGUE = "bot-adversarial-dialogue"
    TOXICHAT = "toxichat"


SELECTED_DATASET = DatasetEnum.PROSOCIAL_DIALOG
SEED = 42
DATASET_DIR = "datasets"
OUT_DIR = "datasets_context_processed"
HATE_LABEL = 1
TRAIN_DATA_MAX = 1000


def norm_text(x: Any) -> str:
    return html.unescape(str(x)).replace("\r\n", "\n").replace("\r", "\n").strip()


def map_app_label(x: Any) -> int:
    return 1 if str(x).strip().lower() == "inappropriate" else 0


def map_offense_label(x: Any) -> int:
    return 0 if str(x).strip().lower() == "safe" else 1


def write_jsonl(rows: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_dataset(dataset_name: str) -> Path:
    dataset_dir = Path(DATASET_DIR)
    for p in dataset_dir.rglob("*"):
        if p.is_dir() and p.name == dataset_name:
            return p
    return None


def get_prosocial_dialog(dataset_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    raw = dataset_path / "raw"
    paths = {
        "train": raw / "prosocial_dialog_train_ab_ko_label.json",
        "valid": raw / "prosocial_dialog_validation_ab_ko_label.json",
        "test": raw / "prosocial_dialog_test_ab_ko_label.json",
    }

    out = {"train": [], "valid": [], "test": []}
    for split, p in paths.items():
        data = json.loads(p.read_text(encoding="utf-8"))
        for dlg in data:
            did = int(dlg["dialogue_id"])
            turns_out = []
            conv_label = 0
            for ti, t in enumerate(dlg["turns"]):
                sp = str(t["speaker"]).strip().upper()
                speaker = 1 if sp == "A" else 2
                lbl = map_app_label(t["label"])
                conv_label = max(conv_label, lbl)

                md = {}
                if "safety_annotation_reasons" in t:
                    md["safety_annotation_reasons"] = t["safety_annotation_reasons"]
                if "rots" in t:
                    md["rots"] = t["rots"]

                turns_out.append(
                    {
                        "turn_idx": ti,
                        "speaker": speaker,
                        "text": norm_text(t["text"]),
                        "label": lbl,
                        "metadata": md,
                    }
                )

            out[split].append(
                {
                    "cid": f"prosocial-dialog_{split}_{did:06d}",
                    "source_dataset": "prosocial-dialog",
                    "split": split,
                    "turns": turns_out,
                    "metadata": {
                        "dialogue_id": did,
                        "conversation_label": int(conv_label),
                    },
                }
            )
    return out


def get_bot_adversarial_dialogue(dataset_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    raw = dataset_path / "raw"
    paths = {
        "train": raw / "train.json",
        "valid": raw / "valid.json",
        "test": raw / "test.json",
    }

    out = {"train": [], "valid": [], "test": []}
    for split, p in paths.items():
        data = json.loads(p.read_text(encoding="utf-8"))
        for dlg in data:
            did = int(dlg["dialogue_id"])
            turns_out = []
            conv_label = 0
            for ti, t in enumerate(dlg["turns"]):
                sp = str(t["speaker"]).strip().upper()
                speaker = 1 if sp == "A" else 2
                lbl = map_app_label(t["label"])
                conv_label = max(conv_label, lbl)

                turns_out.append(
                    {
                        "turn_idx": ti,
                        "speaker": speaker,
                        "text": norm_text(t["text"]),
                        "label": lbl,
                        "metadata": {},
                    }
                )

            out[split].append(
                {
                    "cid": f"bot-adversarial-dialogue_{split}_{did:06d}",
                    "source_dataset": "bot-adversarial-dialogue",
                    "split": split,
                    "turns": turns_out,
                    "metadata": {
                        "dialogue_id": did,
                        "conversation_label": int(conv_label),
                    },
                }
            )
    return out


def get_toxichat(dataset_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    raw = dataset_path / "raw"
    paths = {
        "train": raw / "train.jsonl",
        "valid": raw / "dev.jsonl",
        "test": raw / "test.jsonl",
    }

    out = {"train": [], "valid": [], "test": []}
    for split, p in paths.items():
        idx = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                base_turns = []
                for t in obj["reddit_thread"]:
                    stance = {
                        k: v
                        for k, v in t.items()
                        if str(k).startswith("stance_label_towards_turn_")
                    }
                    base_turns.append(
                        {
                            "turn_idx": len(base_turns),
                            "speaker": 1,
                            "text": norm_text(t["text"]),
                            "label": map_offense_label(t["offense_label"]),
                            "metadata": {
                                "turn_id": t["turn_id"],
                                "offense_targets": t["offense_targets"],
                                "stance": stance,
                            },
                        }
                    )

                for key in ["final_dgpt_response", "final_gpt3_response"]:
                    resp = obj[key]
                    model = resp["chatbot_model"]
                    rstance = {
                        k: v
                        for k, v in resp.items()
                        if str(k).startswith("stance_label_towards_turn_")
                    }

                    turns = list(base_turns)
                    turns.append(
                        {
                            "turn_idx": len(turns),
                            "speaker": 2,
                            "text": norm_text(resp["text"]),
                            "label": map_offense_label(resp["offense_label"]),
                            "metadata": {
                                "turn_id": resp["turn_id"],
                                "chatbot_model": model,
                                "coherence": resp["coherence"],
                                "offense_targets": resp["offense_targets"],
                                "stance": rstance,
                            },
                        }
                    )

                    out[split].append(
                        {
                            "cid": f"toxichat_{split}_{idx:06d}#{model}",
                            "source_dataset": "toxichat",
                            "split": split,
                            "turns": turns,
                            "metadata": {
                                "variant": model,
                                "conversation_label": int(turns[-1]["label"]),
                            },
                        }
                    )

                idx += 1
    return out


DATASET_REGISTRY = {
    DatasetEnum.PROSOCIAL_DIALOG: get_prosocial_dialog,
    DatasetEnum.BOT_ADVERSARIAL_DIALOGUE: get_bot_adversarial_dialogue,
    DatasetEnum.TOXICHAT: get_toxichat,
}


def save_split_count_bar(counts: Dict[str, int], out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    splits = ["train", "valid", "test"]
    vals = [counts.get(s, 0) for s in splits]

    plt.figure(figsize=(7, 4))
    plt.bar(splits, vals)
    plt.title(title)
    plt.ylabel("Num samples")
    for i, v in enumerate(vals):
        plt.text(i, v, str(v), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def apply_train_cap(
    results: Dict[str, List[Dict[str, Any]]], cap: int, seed: int
) -> Dict[str, List[Dict[str, Any]]]:
    tr = results["train"]
    va = results["valid"]
    te = results["test"]

    ntr = len(tr)
    if cap <= 0 or ntr <= cap:
        return results

    r_valid = (len(va) / ntr) if ntr > 0 else 0.0
    r_test = (len(te) / ntr) if ntr > 0 else 0.0

    new_tr_n = cap
    new_va_n = int(round(r_valid * new_tr_n))
    new_te_n = int(round(r_test * new_tr_n))

    rng = random.Random(seed)

    tr_idx = list(range(len(tr)))
    va_idx = list(range(len(va)))
    te_idx = list(range(len(te)))

    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    rng.shuffle(te_idx)

    results["train"] = [tr[i] for i in tr_idx[:new_tr_n]]
    results["valid"] = [va[i] for i in va_idx[:new_va_n]]
    results["test"] = [te[i] for i in te_idx[:new_te_n]]

    return results


def main():
    dataset_path = find_dataset(SELECTED_DATASET.value)
    results = DATASET_REGISTRY[SELECTED_DATASET](dataset_path)
    results = apply_train_cap(results, TRAIN_DATA_MAX, SEED)

    out_root = Path(OUT_DIR) / SELECTED_DATASET.value
    out_root.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split in ["train", "valid", "test"]:
        rows = results[split]
        counts[split] = len(rows)
        print(f"{SELECTED_DATASET.value} {split}: {len(rows)}")
        write_jsonl(rows, out_root / f"{split}.jsonl")

    save_split_count_bar(
        counts,
        out_root / "split_counts.png",
        title=f"{SELECTED_DATASET.value} split counts",
    )


if __name__ == "__main__":
    main()
