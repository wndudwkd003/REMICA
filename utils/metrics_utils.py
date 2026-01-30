# utils/metrics_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from config.config import Config, DatasetEnum


def compute_binary_metrics(
    y_true: List[int],
    y_pred: List[int],
) -> Dict[str, Any]:
    # (기존 그대로)
    if len(y_true) == 0:
        raise ValueError("y_true is empty")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}"
        )

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    n_samples = int(len(y_true_arr))

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="binary",
        zero_division=0,
    )
    precision = float(precision)
    recall = float(recall)
    f1 = float(f1)

    tn, fp, fn, tp = confusion_matrix(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
    ).ravel()

    roc_auc = None
    roc_fpr = None
    roc_tpr = None
    try:
        roc_auc_val = roc_auc_score(y_true_arr, y_pred_arr)
        fpr, tpr, _ = roc_curve(y_true_arr, y_pred_arr)
        roc_auc = float(roc_auc_val)
        roc_fpr = [float(x) for x in fpr]
        roc_tpr = [float(x) for x in tpr]
    except Exception:
        roc_auc = None
        roc_fpr = None
        roc_tpr = None

    return {
        "n_samples": n_samples,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
    }


def save_metrics_csv_and_plots(
    config: Config,
    dataset: DatasetEnum,
    out_dir: Path,
    metrics: Dict[str, Any],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{dataset.name}_{config.api_model.name}_metrics.csv"

    scalar_items = []
    for k, v in metrics.items():
        if isinstance(v, (list, dict)) or v is None:
            continue
        scalar_items.append((k, v))

    lines = ["metric,value"]
    for k, v in scalar_items:
        if isinstance(v, (int, np.integer)):
            lines.append(f"{k},{int(v)}")
        else:
            lines.append(f"{k},{v:.5f}")

    csv_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[metrics_utils] Saved metrics CSV: {csv_path}")

    roc_fpr = metrics.get("roc_fpr")
    roc_tpr = metrics.get("roc_tpr")
    roc_auc = metrics.get("roc_auc", None)

    if roc_fpr is not None and roc_tpr is not None and len(roc_fpr) > 1:
        plt.figure()
        plt.plot(roc_fpr, roc_tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {dataset.name} ({config.api_model.name})")
        plt.legend(loc="lower right")

        roc_path = out_dir / f"{dataset.name}_{config.api_model.name}_roc.png"
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
        print(f"[metrics_utils] Saved ROC plot: {roc_path}")
    else:
        print(f"[metrics_utils] ROC curve not available for {dataset.name} ({out_dir})")
