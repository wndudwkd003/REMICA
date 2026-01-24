# utils/llm_eval.py

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm.auto import tqdm

ANSWER_RE = re.compile(r"<answer>\s*([01])\s*</answer>", re.IGNORECASE)


def parse_answer(text: str) -> Optional[int]:
    """
    생성된 텍스트에서 <answer>0</answer> 또는 <answer>1</answer> 패턴을 파싱.
    못 찾으면 None 리턴.
    """
    m = ANSWER_RE.search(text)
    if not m:
        return None
    try:
        v = int(m.group(1))
        if v in (0, 1):
            return v
    except Exception:
        pass
    return None


def _binary_f1(y_true: List[int], y_pred: List[int]) -> float:
    """
    sklearn 없이 간단한 binary F1 계산 (positive=1 기준).
    """
    assert len(y_true) == len(y_pred)
    tp = fp = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yp == 1 and yt == 1:
            tp += 1
        elif yp == 1 and yt == 0:
            fp += 1
        elif yp == 0 and yt == 1:
            fn += 1

    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@torch.no_grad()
def eval_llm_classifier(
    model: torch.nn.Module,
    tokenizer,
    dataset,
    device: torch.device,
    *,
    max_len: int,
    batch_size: int = 8,
    gen_max_new_tokens: int = 128,
    out_dir: Path,
    desc: str = "eval-llm",
) -> Dict[str, Any]:
    """
    Rem2LMDataset 같은 dataset을 대상으로:
      - prompt만 넣어서 generate
      - <answer>...</answer> 파싱
      - acc / F1 계산
      - CSV/JSON/PNG 저장
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    all_true: List[int] = []
    all_pred: List[Optional[int]] = []
    rows: List[Dict[str, Any]] = []

    n = len(dataset)
    pbar = tqdm(range(0, n, batch_size), desc=desc)

    for start in pbar:
        end = min(n, start + batch_size)
        batch = [dataset[i] for i in range(start, end)]

        prompts = [b["prompt"] for b in batch]
        labels = [int(b["label"]) for b in batch]
        sids = [b.get("sid") for b in batch]

        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=gen_max_new_tokens,
            do_sample=False,
        )

        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for sid, prompt, text, y in zip(sids, prompts, texts, labels):
            pred = parse_answer(text)
            all_true.append(y)
            all_pred.append(pred)

            rows.append(
                {
                    "sid": sid,
                    "label": y,
                    "pred": pred,
                    "parsed_ok": pred in (0, 1),
                    "prompt": prompt,
                    "output": text,
                }
            )

    # None / 이상값 제거 후 metrics 계산
    valid_idx = [i for i, p in enumerate(all_pred) if p in (0, 1)]
    if len(valid_idx) == 0:
        acc = 0.0
        f1 = 0.0
    else:
        y_true = [all_true[i] for i in valid_idx]
        y_pred = [int(all_pred[i]) for i in valid_idx]
        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        acc = correct / len(y_true)
        f1 = _binary_f1(y_true, y_pred)

    parsed_ratio = len(valid_idx) / len(all_true) if all_true else 0.0

    # CSV 저장
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "llm_eval_details.csv", index=False)

    # JSON 저장
    metrics = {
        "acc": acc,
        "f1": f1,
        "parsed_ratio": parsed_ratio,
        "n_total": len(all_true),
        "n_parsed": len(valid_idx),
    }
    with open(out_dir / "llm_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 간단한 bar plot (acc / f1)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["acc", "f1"], [acc, f1])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("LLM Classification Metrics")
    for i, v in enumerate([acc, f1]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "llm_eval_metrics.png")
    plt.close(fig)

    return metrics
