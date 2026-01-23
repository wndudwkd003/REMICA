# utils/viz_utils.py

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_cross_grid(
    csv_path: str | Path,
    out_path: str | Path,
    metric: str = "acc",  # "acc" | "loss" | "n"
    dataset_order: Optional[List[str]] = None,
    title: Optional[str] = None,
    fmt: str = ".3f",  # cell text format
):
    """
    cross_test.csv를 (train_dataset x test_dataset) 매트릭스로 pivot 후
    heatmap 형태로 저장합니다.

    csv columns required:
      - train_dataset
      - test_dataset
      - acc / loss / n
    """
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        raise ValueError(f"metric='{metric}' not in csv columns: {list(df.columns)}")

    mat = df.pivot(index="train_dataset", columns="test_dataset", values=metric)

    # 지정된 순서가 있으면 그 순서대로 정렬
    if dataset_order is not None:
        mat = mat.reindex(index=dataset_order, columns=dataset_order)

    values = mat.to_numpy(dtype=float)
    rows = list(mat.index)
    cols = list(mat.columns)

    # 그림
    fig, ax = plt.subplots(figsize=(1.2 * max(len(cols), 5), 1.0 * max(len(rows), 4)))
    im = ax.imshow(values, aspect="auto")  # colormap 기본값 사용

    # 축 레이블
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(rows)

    ax.set_xlabel("Test dataset")
    ax.set_ylabel("Train dataset")
    ax.set_title(title or f"Cross-test grid ({metric})")

    # 셀 값 표기
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = values[i, j]
            if np.isnan(v):
                s = "NA"
            else:
                s = format(v, fmt)
            ax.text(j, i, s, ha="center", va="center", fontsize=9)

    # 컬러바
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_train_valid_curves(
    history_csv: Path, out_path: Path, metric: str = "loss", title: str = ""
):
    """
    history_csv columns 예:
      epoch, tr_loss, tr_acc, va_loss, va_acc
    metric: "loss" | "acc"
    """
    df = pd.read_csv(history_csv)

    x = df["epoch"].tolist()

    if metric == "loss":
        y_tr = df["tr_loss"].tolist()
        y_va = df["va_loss"].tolist()
        ylab = "Loss"
        if not title:
            title = "Train/Valid Loss"
    elif metric == "acc":
        y_tr = df["tr_acc"].tolist()
        y_va = df["va_acc"].tolist()
        ylab = "Accuracy"
        if not title:
            title = "Train/Valid Accuracy"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    plt.figure()
    plt.plot(x, y_tr, label="train")
    plt.plot(x, y_va, label="valid")
    plt.xlabel("Epoch")
    plt.ylabel(ylab)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_1xN_grid(
    values: List[float],
    col_names: List[str],
    out_path: Path | str,
    title: str,
    fmt: str = ".3f",
):
    """
    1 x N 형태의 heatmap (GPT vs 여러 test dataset) 그릴 때 사용.
    (gpt_infer 전용이었는데, 여기로 뺀 버전)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(values, dtype=np.float32).reshape(1, -1)
    fig = plt.figure(figsize=(max(8, len(col_names) * 1.1), 2.2))
    ax = plt.gca()
    ax.imshow(data, aspect="auto")  # 색상 지정 X

    ax.set_title(title)
    ax.set_yticks([0])
    ax.set_yticklabels(["GPT"])
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right")

    for j, v in enumerate(values):
        s = f"{v:{fmt}}" if v is not None else "NA"
        ax.text(j, 0, s, ha="center", va="center", color="black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
