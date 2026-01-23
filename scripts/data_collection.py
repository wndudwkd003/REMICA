import json
import os
import random
from collections import Counter
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.filter_utils import is_only_url_or_symbol


class DatasetEnum(Enum):
    DiaSafety = "DiaSafety"
    GabHate = "gab_hate"
    HSOL = "hate-speech-and-offensive-language-master"
    HateXplain = "hatexplain"
    RealToxicityPrompts = "real-toxicity-prompts"
    OffenseEval = "offenseval"
    HSD = "hate-speech-dataset-master"
    ToxiGen = "toxigen"
    ToxiSpanSE = "ToxiSpanSE"
    ToxiCR = "toxicr"
    HSDCD = "hsdcd"
    ISHate = "ISHate"


SELECT_DATA = DatasetEnum.HSDCD
DATASET_DIR = "datasets"
DO_MODE = "process_data"  # "check_count" | "process_data"
SEED = 42
OUT_DIR = "datasets_processed"
HATE_LABEL = 1

COUNT_MATCHING = True
MAX_TRAIN_SAMPLES: int | None = 8000  # train split의 최대 샘플 수 (valid/test는 8:1:1 비율로 자동 결정)



os.makedirs(OUT_DIR, exist_ok=True)

def limit_max_samples(results, max_train: int | None, seed: int):
    """
    train/valid/test 전체 크기를 줄이되,
    - 최종적으로 train : valid : test ≈ 8 : 1 : 1 비율 유지
    - 각 split 내부에서는 label(0/1) 균형 유지
    max_train: train split의 최대 샘플 수 (None이면 제한 없음)
    """
    if max_train is None:
        return results

    rng = random.Random(seed + 123)

    # 현재 split별 샘플 수
    n_train = len(results["train"])
    n_valid = len(results["valid"])
    n_test = len(results["test"])

    # 1) train 타겟 개수 (상한)
    target_train = min(n_train, max_train)

    # 2) 8 : 1 : 1 비율 기준 valid/test 타겟 개수
    #    train : valid = 8 : 1  → valid ≈ train / 8
    #    train : test  = 8 : 1  → test  ≈ train / 8
    base_valid = target_train // 8
    if base_valid < 1:
        base_valid = 1  # 최소 1개는 유지

    target_valid = min(n_valid, base_valid)
    target_test = min(n_test, base_valid)

    targets = {
        "train": target_train,
        "valid": target_valid,
        "test": target_test,
    }

    out = {"train": [], "valid": [], "test": []}

    for split in ["train", "valid", "test"]:
        items = list(results[split])
        target = targets[split]

        # 이미 target 이하이면 그대로 사용
        if len(items) <= target:
            out[split] = items
            continue

        # 라벨별 그룹화
        by_label: dict[int, list[dict]] = {}
        for it in items:
            lbl = it["label"]
            by_label.setdefault(lbl, []).append(it)

        labels = sorted(by_label.keys())

        # binary label일 때: 라벨 균형 유지하며 target 개수만큼 샘플링
        if len(labels) == 2 and 0 in by_label and 1 in by_label:
            zeros = by_label[0]
            ones = by_label[1]

            rng.shuffle(zeros)
            rng.shuffle(ones)

            # 타겟을 반반 나누되, 한 클래스가 부족하면 다른 쪽에서 보충
            half = target // 2
            rem = target - 2 * half  # 0 또는 1

            take0 = min(len(zeros), half + rem)
            take1 = min(len(ones), target - take0)

            # 한쪽이 부족하면 남는 쪽에서 추가 확보
            if take0 + take1 < target:
                # zeros에서 더 뽑을 수 있으면 먼저 시도
                extra0 = min(len(zeros) - take0, target - (take0 + take1))
                take0 += max(extra0, 0)

            if take0 + take1 < target:
                # 그래도 모자라면 ones에서 추가
                extra1 = min(len(ones) - take1, target - (take0 + take1))
                take1 += max(extra1, 0)

            selected = zeros[:take0] + ones[:take1]
            rng.shuffle(selected)
            out[split] = selected

        else:
            # 라벨 종류가 2개가 아닌 경우: 그냥 랜덤으로 target 개수 자르기
            rng.shuffle(items)
            out[split] = items[:target]

    return out

def find_dataset(dataset_name: str):
    dataset_dir = Path(DATASET_DIR)
    for dataset_path in dataset_dir.rglob("*"):
        if dataset_path.is_dir() and dataset_path.name == dataset_name:
            return dataset_path
    return None


def get_ishate(dataset_path: Path, dataset_name: str):
    results = {"train": [], "valid": [], "test": []}

    base_dir = dataset_path / "decompressed"
    train_path = base_dir / "ishate_train.csv"
    valid_path = base_dir / "ishate_dev.csv"
    test_path = base_dir / "ishate_test.csv"

    train_df = pd.read_csv(train_path, dtype=str)
    valid_df = pd.read_csv(valid_path, dtype=str)
    test_df = pd.read_csv(test_path, dtype=str)

    def to_items(df, split_name: str):
        items = []
        for i, row in df.iterrows():
            text = str(row.get("cleaned_text", "")).replace("\n", " ")
            text = " ".join(text.split())
            if not text or is_only_url_or_symbol(text):
                continue

            hateful_layer = str(row.get("hateful_layer", "")).strip()
            label = 1 if hateful_layer == "HS" else 0

            items.append(
                {
                    "id": f"{dataset_name}_{split_name}_{i}",
                    "text": text,
                    "label": label,
                    "metadata": {
                        "message_id": row.get("message_id", None),
                        "source": row.get("source", None),
                        "hateful_layer": row.get("hateful_layer", None),
                        "implicit_layer": row.get("implicit_layer", None),
                        "subtlety_layer": row.get("subtlety_layer", None),
                        "implicit_props_layer": row.get("implicit_props_layer", None),
                        "target": row.get("target", None),
                    },
                }
            )
        return items

    results["train"] = to_items(train_df, "train")
    results["valid"] = to_items(valid_df, "valid")
    results["test"] = to_items(test_df, "test")

    return results


def get_hsdcd(dataset_path: Path, dataset_name: str):
    results = {"train": [], "valid": [], "test": []}

    csv_path = dataset_path / "archive" / "HateSpeechDatasetBalanced.csv"
    if csv_path.exists() is False:
        csv_path = dataset_path / "archive" / "HateSpeechDataset.csv"

    df = pd.read_csv(csv_path)

    df["text"] = df["Content"].astype(str)
    df["label"] = df["Label"].astype(int)

    df = df[~df["text"].apply(is_only_url_or_symbol)]

    train_df, tmp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    valid_df, test_df = train_test_split(
        tmp_df, test_size=0.5, random_state=SEED, stratify=tmp_df["label"]
    )

    def to_items(df_split, split_name: str):
        items = []
        for idx, row in df_split.iterrows():
            text = str(row["text"]).replace("\n", " ")
            text = " ".join(text.split())
            items.append(
                {
                    "id": f"{dataset_name}_{split_name}_{idx}",
                    "text": text,
                    "label": int(row["label"]),
                    "metadata": {},
                }
            )
        return items

    results["train"] = to_items(train_df, "train")
    results["valid"] = to_items(valid_df, "valid")
    results["test"] = to_items(test_df, "test")

    return results


def get_toxicr(dataset_path: Path, dataset_name: str):
    results = {"train": [], "valid": [], "test": []}

    xlsx_path = dataset_path / "toxicr.xlsx"
    df = pd.read_excel(xlsx_path)

    label_col = [c for c in df.columns if str(c).lower().startswith("is_to")][0]

    df["text"] = df["message"].astype(str)
    df["label"] = df[label_col].astype(int)

    df = df[~df["text"].apply(is_only_url_or_symbol)]

    train_df, tmp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    valid_df, test_df = train_test_split(
        tmp_df, test_size=0.5, random_state=SEED, stratify=tmp_df["label"]
    )

    def to_items(df_split, split_name: str):
        items = []
        for idx, row in df_split.iterrows():
            text = str(row["text"]).replace("\n", " ")
            text = " ".join(text.split())
            items.append(
                {
                    "id": f"{dataset_name}_{split_name}_{idx}",
                    "text": text,
                    "label": int(row["label"]),
                    "metadata": {},
                }
            )
        return items

    results["train"] = to_items(train_df, "train")
    results["valid"] = to_items(valid_df, "valid")
    results["test"] = to_items(test_df, "test")

    return results


def get_toxispanse(dataset_path: Path, dataset_name: str):
    results = {"train": [], "valid": [], "test": []}

    csv_path = dataset_path / "CR_full_span_dataset.csv"
    df = pd.read_csv(csv_path)

    df["is_toxic"] = df["is_toxic"].fillna(0)
    df["label"] = df["is_toxic"].astype(int)

    df["text"] = df["text"].astype(str)
    df = df[~df["text"].apply(is_only_url_or_symbol)]

    train_df, tmp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    valid_df, test_df = train_test_split(
        tmp_df, test_size=0.5, random_state=SEED, stratify=tmp_df["label"]
    )

    def extract_span_words(row):
        s = row.get("final_selected_text", "")
        s = "" if pd.isna(s) else str(s)
        s = s.replace("\n", " ").strip()

        if not s:
            a = row.get("rater1_text", "")
            b = row.get("rater2_text", "")
            a = "" if pd.isna(a) else str(a)
            b = "" if pd.isna(b) else str(b)
            s = ",".join([x for x in [a.strip(), b.strip()] if x])

        s = " ".join(s.split())
        if not s:
            return []

        raw = s.replace("/", " ").replace("-", " ").replace("_", " ")
        raw = raw.translate(str.maketrans({c: " " for c in "[]{}()\"'.,;:!?,"}))
        toks = [t.strip() for t in raw.split() if t.strip()]

        seen = set()
        uniq = []
        for t in toks:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        return uniq

    def to_items(df_split, split_name: str):
        items = []
        for _, row in df_split.iterrows():
            text = str(row["text"]).replace("\n", " ")
            text = " ".join(text.split())

            items.append(
                {
                    "id": f"{dataset_name}_{split_name}_{row['id']}",
                    "text": text,
                    "label": int(row["label"]),
                    "metadata": {
                        "spans": extract_span_words(row),
                        "is_conflict": row.get("is_conflict", None),
                        "final_selected_text": row.get("final_selected_text", None),
                        "rater1_text": row.get("rater1_text", None),
                        "rater2_text": row.get("rater2_text", None),
                    },
                }
            )
        return items

    results["train"] = to_items(train_df, "train")
    results["valid"] = to_items(valid_df, "valid")
    results["test"] = to_items(test_df, "test")

    return results


def get_toxigen_annotated_only(dataset_path: Path, dataset_name: str):
    results = {"train": [], "valid": [], "test": []}

    train_path = dataset_path / "annotated" / "train.jsonl"
    test_path = dataset_path / "annotated" / "test.jsonl"

    def read_jsonl(p: Path):
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(test_path)

    def to_items(rows, split_name: str):
        items = []
        for i, r in enumerate(rows):
            text = str(r.get("text", "")).replace("\n", " ")
            text = " ".join(text.split())
            if not text or is_only_url_or_symbol(text):
                continue

            y = r["toxicity_human"]
            label = int(y >= 3.0)

            items.append(
                {
                    "id": f"{dataset_name}_{split_name}_{i}",
                    "text": text,
                    "label": label,
                    "metadata": {
                        "target_group": r.get("target_group", None),
                        "factual": r.get("factual?", None),
                        "ingroup_effect": r.get("ingroup_effect", None),
                        "lewd": r.get("lewd", None),
                        "framing": r.get("framing", None),
                        "predicted_group": r.get("predicted_group", None),
                        "stereotyping": r.get("stereotyping", None),
                        "intent": r.get("intent", None),
                        "toxicity_ai": r.get("toxicity_ai", None),
                        "toxicity_human": r.get("toxicity_human", None),
                        "predicted_author": r.get("predicted_author", None),
                        "actual_method": r.get("actual_method", None),
                    },
                }
            )
        return items

    all_train_items = to_items(train_rows, "train_full")
    tr_items, va_items = train_test_split(
        all_train_items,
        test_size=0.2,
        random_state=SEED,
        stratify=[x["label"] for x in all_train_items],
    )

    results["train"] = [
        {
            **it,
            "id": it["id"].replace(
                f"{dataset_name}_train_full_", f"{dataset_name}_train_"
            ),
        }
        for it in tr_items
    ]
    results["valid"] = [
        {
            **it,
            "id": it["id"].replace(
                f"{dataset_name}_train_full_", f"{dataset_name}_valid_"
            ),
        }
        for it in va_items
    ]
    results["test"] = to_items(test_rows, "test")

    return results


def get_hate_speech_dataset_master(dataset_path: Path, dataset_name: str):
    results = {"train": [], "valid": [], "test": []}

    meta_path = dataset_path / "raw" / "annotations_metadata.csv"
    df = pd.read_csv(meta_path, dtype=str)

    df["label"] = df["label"].astype(str)
    df["label_bin"] = (df["label"] == "hate").astype(int)

    train_dir = dataset_path / "raw" / "sampled_train"
    test_dir = dataset_path / "raw" / "sampled_test"

    train_ids = {p.stem for p in train_dir.glob("*.txt")}
    test_ids = {p.stem for p in test_dir.glob("*.txt")}

    train_df = df[df["file_id"].isin(train_ids)].copy()
    test_df = df[df["file_id"].isin(test_ids)].copy()

    tr_df, va_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=SEED,
        stratify=train_df["label_bin"],
    )

    def to_items(df_split, split_name, base_dir: Path):
        items = []
        for _, row in df_split.iterrows():
            file_id = row["file_id"]
            text = (base_dir / f"{file_id}.txt").read_text(encoding="utf-8")
            text = " ".join(text.replace("\n", " ").split())
            if is_only_url_or_symbol(text):
                continue
            items.append(
                {
                    "id": f"{dataset_name}_{split_name}_{file_id}",
                    "text": text,
                    "label": int(row["label_bin"]),
                    "metadata": {},
                }
            )
        return items

    results["train"] = to_items(tr_df, "train", train_dir)
    results["valid"] = to_items(va_df, "valid", train_dir)
    results["test"] = to_items(test_df, "test", test_dir)

    return results


def get_offenseval(dataset_path: Path, dataset_name: str):
    results = {"train": [], "valid": [], "test": []}

    train_path = dataset_path / "raw" / "data" / "olid-training-v1.0.tsv"
    train_df = pd.read_csv(train_path, sep="\t", dtype=str)

    train_df["text"] = train_df["tweet"].astype(str)
    train_df = train_df[~train_df["text"].apply(is_only_url_or_symbol)]
    train_df["label"] = (train_df["subtask_a"] == "OFF").astype(int)

    tr_df, va_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=SEED,
        stratify=train_df["label"],
    )

    def to_items(df_split, split_name):
        items = []
        for _, row in df_split.iterrows():
            items.append(
                {
                    "id": f"{dataset_name}_{split_name}_{row['id']}",
                    "text": row["text"],
                    "label": int(row["label"]),
                    "metadata": {},
                }
            )
        return items

    results["train"] = to_items(tr_df, "train")
    results["valid"] = to_items(va_df, "valid")

    test_path = dataset_path / "raw" / "data" / "testset-levela.tsv"
    test_labels_path = dataset_path / "raw" / "data" / "labels-levela.csv"

    test_df = pd.read_csv(test_path, sep="\t", dtype=str)
    test_df["text"] = test_df["tweet"].astype(str)
    test_df = test_df[~test_df["text"].apply(is_only_url_or_symbol)]

    labels_df = pd.read_csv(
        test_labels_path, header=None, names=["id", "subtask_a"], dtype=str
    )
    merged = test_df.merge(labels_df, on="id", how="inner")
    merged["label"] = (merged["subtask_a"] == "OFF").astype(int)

    results["test"] = [
        {
            "id": f"{dataset_name}_test_{row['id']}",
            "text": row["text"],
            "label": int(row["label"]),
            "metadata": {},
        }
        for _, row in merged.iterrows()
    ]

    return results


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
    DatasetEnum.DiaSafety: get_diasafety,
    DatasetEnum.GabHate: get_gabhate,
    DatasetEnum.HSOL: get_hsol,
    DatasetEnum.HateXplain: get_hatexplain,
    DatasetEnum.RealToxicityPrompts: get_real_toxicity_prompts,
    DatasetEnum.OffenseEval: get_offenseval,
    DatasetEnum.HSD: get_hate_speech_dataset_master,
    DatasetEnum.ToxiGen: get_toxigen_annotated_only,
    DatasetEnum.ToxiSpanSE: get_toxispanse,
    DatasetEnum.ToxiCR: get_toxicr,
    DatasetEnum.HSDCD: get_hsdcd,
    DatasetEnum.ISHate: get_ishate,
}


def match_label_counts(results, seed: int):
    rng = random.Random(seed)
    out = {"train": [], "valid": [], "test": []}

    for split in ["train", "valid", "test"]:
        items = list(results[split])
        labels = [it["label"] for it in items]
        cnt = Counter(labels)

        if len(cnt) < 2:
            out[split] = items
            continue

        n0 = cnt[0]
        n1 = cnt[1]
        target = min(n0, n1)

        zeros = [it for it in items if it["label"] == 0]
        ones = [it for it in items if it["label"] == 1]

        rng.shuffle(zeros)
        rng.shuffle(ones)

        kept = zeros[:target] + ones[:target]
        rng.shuffle(kept)

        out[split] = kept

    return out


def do_check_count(results):
    print(f"DO_MODE: {DO_MODE} (Only showing split counts)")

    print("== Raw counts ==")
    for split in ["train", "valid", "test"]:
        print(f"{split}: {len(results[split])} samples")

    if COUNT_MATCHING:
        results = match_label_counts(results, seed=SEED)
        print("== After Count Matching ==")
        for split in ["train", "valid", "test"]:
            print(f"{split}: {len(results[split])} samples")

    if MAX_TRAIN_SAMPLES is not None:
        results = limit_max_samples(results, MAX_TRAIN_SAMPLES, seed=SEED)
        print(f"== After MAX_TRAIN_SAMPLES={MAX_TRAIN_SAMPLES} with 8:1:1 ratio ==")
        for split in ["train", "valid", "test"]:
            print(f"{split}: {len(results[split])} samples")


def do_process_data(results, output_path: Path):
    print(f"DO_MODE: {DO_MODE} (Saving processed JSONL files)")

    if COUNT_MATCHING:
        results = match_label_counts(results, seed=SEED)

    if MAX_TRAIN_SAMPLES is not None:
        results = limit_max_samples(results, MAX_TRAIN_SAMPLES, seed=SEED)

    output_path.mkdir(parents=True, exist_ok=True)
    save_jsonl(results, output_path)
    print(f"Saved processed data to: {output_path}")

    # 라벨 분포도 "최종 결과" 기준으로 그림
    output_path = Path(OUT_DIR) / SELECT_DATA.name
    draw_label_distribution(results, output_path)


if __name__ == "__main__":
    dataset_path = find_dataset(SELECT_DATA.value)
    if dataset_path is None:
        print(f"Dataset '{SELECT_DATA.value}' not found under {DATASET_DIR}")
        exit(1)

    print(f"Loading {SELECT_DATA.name} dataset from: {dataset_path}")
    results = DATASET_REGISTRY[SELECT_DATA](dataset_path, SELECT_DATA.name)

    if DO_MODE == "check_count":
        do_check_count(results)

    elif DO_MODE == "process_data":
        do_process_data(results, Path(OUT_DIR) / SELECT_DATA.name)

    else:
        print(f"Unknown DO_MODE: {DO_MODE}")
