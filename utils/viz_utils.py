# utils/viz_utils.py


from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    im = ax.imshow(values, aspect="auto")  # colormap은 기본값 사용(색 지정 금지)

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
