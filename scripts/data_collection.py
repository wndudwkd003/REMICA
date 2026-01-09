import os
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from scripts.filter_utils import is_only_url_or_symbol
from collections import Counter

import matplotlib.pyplot as plt

DATASET_NAME_MAP = {
    "DiaSafety": "DiaSafety",
    "gab_hate": "GabHate",
    "hate-speech-and-offensive-language-master": "HSOL",
    "hatexplain": "HateXplain",
    "real-toxicity-prompts": "RealToxicityPrompts",
}

SELECT_DATA = "real-toxicity-prompts"  # DiaSafety | gab_hate | "hate-speech-and-offensive-language-master" | "hatexplain" | "real-toxicity-prompts"
DATASET_DIR = "datasets"
DO_MODE = "process_data"  # "check_count" | "process_data"
SEED = 42
OUT_DIR = "datasets_processed"
HATE_LABEL = 1
os.makedirs(OUT_DIR, exist_ok=True)


def find_dataset(dataset_name: str):
    dataset_dir = Path(DATASET_DIR)
    for dataset_path in dataset_dir.rglob("*"):
        if dataset_path.is_dir() and dataset_path.name == dataset_name:
            return dataset_path
    return None


def get_real_toxicity_prompts(
    dataset_path: Path,
    dataset_name: str,
    toxicity_threshold: float = 0.5,
):
    results = {"train": [], "valid": [], "test": []}
    split_map = {"train": "train", "valid": "valid", "test": "test"}

    score_keys = [
        "toxicity",
        "profanity",
        "sexually_explicit",
        "flirtation",
        "identity_attack",
        "threat",
        "insult",
        "severe_toxicity",
    ]

    def add_part(split_name: str, idx: int, part: str, obj: dict):
        # 필수 키(text + scores)가 없거나 None이면 스킵
        if obj["text"] is None:
            return
        for k in score_keys:
            if obj[k] is None:
                return

        toxicity = obj["toxicity"]
        label = int(toxicity >= toxicity_threshold)

        results[split_name].append(
            {
                "id": f"{dataset_name}_{split_name}_{idx}_{part}",
                "text": obj["text"],
                "label": label,
                "metadata": {k: obj[k] for k in score_keys},
            }
        )

    for split_file, split_name in split_map.items():
        split_path = dataset_path / "raw" / f"{split_file}.json"
        if not split_path.exists():
            print(f"Warning: {split_path} not found, skipping.")
            continue

        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for i, item in enumerate(data):
            add_part(split_name, i, "prompt", item["prompt"])
            add_part(split_name, i, "cont", item["continuation"])

    return results


def get_hatexplain(dataset_path: Path, dataset_name: str):
    results = {"train": [], "valid": [], "test": []}
    split_map = {"train": "train", "validation": "valid", "test": "test"}

    for split_file, split_name in split_map.items():
        split_path = dataset_path / "raw" / f"{split_file}.json"
        if not split_path.exists():
            continue

        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for i, item in enumerate(data):
            tokens = item["post_tokens"]
            annot_labels = item["annotators"]["label"]
            raw_targets = item["annotators"].get("target", [])
            rationales_list = item.get("rationales", [])

            # ===== 텍스트 클렌징 =====
            clean_tokens = [
                tok
                for tok in tokens
                if tok not in {"<user>", "<url>", "<number>", "<censored>"}
            ]
            text = " ".join(clean_tokens).replace("\n", " ")
            text = " ".join(text.split())

            # ===== 다수결 라벨 (0 or 2 → 부적절) =====
            majority = Counter(annot_labels).most_common(1)[0][0]
            label = int(majority in [0, 2])

            # ===== 타겟 =====
            targets = sorted({t for sub in raw_targets for t in sub if t != "None"})

            # ===== 근거 토큰 추출 =====
            rationale_token_set = set()
            for rationale, ann_label in zip(rationales_list, annot_labels):
                if ann_label not in [0, 2]:  # only hatespeech or offensive
                    continue

                for tok, r in zip(tokens, rationale):
                    if r == 1 and tok not in {
                        "<user>",
                        "<url>",
                        "<number>",
                        "<censored>",
                    }:
                        rationale_token_set.add(tok)

            rationale_tokens = sorted(rationale_token_set)

            # ===== 라벨 분포 =====
            count_dict = Counter(annot_labels)
            label_counts = {
                "hatespeech": count_dict.get(0, 0),
                "normal": count_dict.get(1, 0),
                "offensive": count_dict.get(2, 0),
            }

            results[split_name].append(
                {
                    "id": f"{dataset_name}_{split_name}_{i}",
                    "text": text,
                    "label": label,
                    "metadata": {
                        "targets": targets,
                        "rationales": rationale_tokens,
                        "label_counts": label_counts,
                    },
                }
            )

    return results


def draw_label_distribution(results, output_dir: Path):
    splits = ["train", "valid", "test"]
    counts = {"train": {}, "valid": {}, "test": {}}

    for split in splits:
        labels = [item["label"] for item in results[split]]
        label_count = Counter(labels)
        counts[split] = label_count

    # 그래프 준비
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [0, 1]
    x = range(len(splits))
    bar_width = 0.35

    for i, label in enumerate(labels):
        values = [counts[split].get(label, 0) for split in splits]
        ax.bar(
            [pos + i * bar_width for pos in x],
            values,
            bar_width,
            label=f"Label {label}",
        )

    ax.set_xticks([pos + bar_width / 2 for pos in x])
    ax.set_xticklabels(splits)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Label Distribution by Split")
    ax.legend()

    plt.tight_layout()

    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "label_distribution.png"
    plt.savefig(out_path)
    print(f"Saved label distribution plot to: {out_path}")
    plt.close()


def get_gabhate(dataset_path: Path, dataset_name: str):
    dataset_path = dataset_path / "raw"
    results = {"train": [], "valid": [], "test": []}

    df = pd.read_csv(dataset_path / "ghc_train.tsv", sep="\t")
    df["label"] = ((df["hd"] == 1) | (df["cv"] == 1) | (df["vo"] == 1)).astype(int)
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=SEED)

    def to_dict_list(df_split, split_name):
        items = []
        for i, row in df_split.iterrows():
            text = str(row["text"])
            if is_only_url_or_symbol(text):
                continue
            items.append(
                {
                    "id": f"{dataset_name}_{split_name}_{i}",
                    "text": text,
                    "label": int(row["label"]),
                    "metadata": {},
                }
            )
        return items

    results["train"] = to_dict_list(train_df, "train")
    results["valid"] = to_dict_list(valid_df, "valid")

    test_path = dataset_path / "ghc_test.tsv"
    if test_path.exists():
        test_df = pd.read_csv(test_path, sep="\t")
        test_df["label"] = (
            (test_df["hd"] == 1) | (test_df["cv"] == 1) | (test_df["vo"] == 1)
        ).astype(int)
        results["test"] = to_dict_list(test_df, "test")
    else:
        print(f"Warning: {test_path} not found, skipping test set.")

    return results


def get_hsol(dataset_path: Path, dataset_name: str):
    csv_path = dataset_path / "raw" / "data" / "labeled_data.csv"
    results = {"train": [], "valid": [], "test": []}

    df = pd.read_csv(csv_path)

    # === 핵심 라벨 정의 ===
    df["label"] = ((df["hate_speech"] > 0) | (df["offensive_language"] > 0)).astype(int)

    df["text"] = df["tweet"].astype(str)

    # 의미 없는 텍스트 제거
    df = df[~df["text"].apply(is_only_url_or_symbol)]

    # split: train / valid / test = 8 / 1 / 1
    train_df, tmp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    valid_df, test_df = train_test_split(
        tmp_df, test_size=0.5, random_state=SEED, stratify=tmp_df["label"]
    )

    def to_dict_list(df_split, split_name):
        return [
            {
                "id": f"{dataset_name}_{split_name}_{i}",
                "text": row["text"],
                "label": int(row["label"]),
                "metadata": {
                    "hate_speech_votes": int(row["hate_speech"]),
                    "offensive_votes": int(row["offensive_language"]),
                    "neither_votes": int(row["neither"]),
                },
            }
            for i, row in df_split.iterrows()
        ]

    results["train"] = to_dict_list(train_df, "train")
    results["valid"] = to_dict_list(valid_df, "valid")
    results["test"] = to_dict_list(test_df, "test")

    return results


def get_diasafety(dataset_path: Path, dataset_name: str):
    targets = ["train", "val", "test"]
    results = {"train": [], "valid": [], "test": []}
    for split in targets:
        split_path = dataset_path / "raw" / f"{split}.json"
        if not split_path.exists():
            print(f"Warning: {split_path} not found, skipping.")
            continue
        with open(split_path, "r") as f:
            data = json.load(f)
        for i, item in enumerate(data):
            results["valid" if split == "val" else split].append(
                {
                    "id": f"{dataset_name}_{split}_{i}",
                    "text": item["response"],
                    "label": HATE_LABEL if item["label"] == "Unsafe" else 0,
                    "metadata": {
                        "context": item["context"],
                        "category": item["category"],
                    },
                }
            )
    return results


def save_jsonl(data, output_path: Path):
    for split, items in data.items():
        seen_texts = set()
        split_output_path = output_path / f"{split}.jsonl"
        with open(split_output_path, "w") as f:
            for item in items:
                if item["text"] in seen_texts:
                    continue
                seen_texts.add(item["text"])
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + "\n")


DATASET_REGISTRY = {
    "DiaSafety": get_diasafety,
    "gab_hate": get_gabhate,
    "hate-speech-and-offensive-language-master": get_hsol,
    "hatexplain": get_hatexplain,
    "real-toxicity-prompts": get_real_toxicity_prompts,
}


def do_check_count(results):
    print(f"DO_MODE: {DO_MODE} (Only showing split counts)")
    for split in ["train", "valid", "test"]:
        print(f"{split}: {len(results[split])} samples")


def do_process_data(results, output_path: Path):
    print(f"DO_MODE: {DO_MODE} (Saving processed JSONL files)")
    output_path.mkdir(parents=True, exist_ok=True)
    save_jsonl(results, output_path)
    print(f"Saved processed data to: {output_path}")

    # 라벨 분포 그래프 저장
    output_path = Path(OUT_DIR) / DATASET_NAME_MAP[SELECT_DATA]
    draw_label_distribution(results, output_path)


if __name__ == "__main__":
    dataset_path = find_dataset(SELECT_DATA)
    if dataset_path is None:
        print(
            f"Dataset '{DATASET_NAME_MAP[SELECT_DATA]}' not found under {DATASET_DIR}"
        )
        exit(1)

    print(f"Loading {DATASET_NAME_MAP[SELECT_DATA]} dataset from: {dataset_path}")
    results = DATASET_REGISTRY[SELECT_DATA](dataset_path, DATASET_NAME_MAP[SELECT_DATA])

    if DO_MODE == "check_count":
        do_check_count(results)

    elif DO_MODE == "process_data":
        do_process_data(results, Path(OUT_DIR) / DATASET_NAME_MAP[SELECT_DATA])

    else:
        print(f"Unknown DO_MODE: {DO_MODE}")
