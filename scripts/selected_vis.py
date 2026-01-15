# make_selected_cross_test.py
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from config.config import DatasetEnum


# =========================
# 1) 여기만 고치세요 (선택)
# =========================
SELECTED: List[DatasetEnum] = [
    DatasetEnum.HSOL,
    DatasetEnum.HateXplain,
    DatasetEnum.DiaSafety,
    DatasetEnum.ToxiSpanSE,
    DatasetEnum.HSD,
]


RUN_DIR = Path("runs/20260111_111016_ModernBERT")
IN_JSON = RUN_DIR / "results" / "cross_test_retest.json"
OUT_DIR = RUN_DIR / "selected"  # 영문 폴더명 요구사항


@dataclass(frozen=True)
class Cell:
    loss: float | None
    acc: float | None
    n: float | None


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def resolve_key(wanted: DatasetEnum, keys: List[str]) -> str | None:
    """
    cross_test_retest.json의 키가 Enum.name일 수도, Enum.value일 수도, 혹은 약간 다를 수도 있으므로
    최대한 안전하게 매칭합니다.
    우선순위:
      1) exact Enum.name
      2) exact Enum.value
      3) case-insensitive match
    """
    if wanted.name in keys:
        return wanted.name
    if wanted.value in keys:
        return wanted.value

    lower_map = {k.lower(): k for k in keys}
    if wanted.name.lower() in lower_map:
        return lower_map[wanted.name.lower()]
    if wanted.value.lower() in lower_map:
        return lower_map[wanted.value.lower()]

    return None


def extract_cell(d: Dict[str, Any]) -> Cell:
    return Cell(
        loss=_safe_float(d.get("loss")),
        acc=_safe_float(d.get("acc")),
        n=_safe_float(d.get("n")),
    )


def main() -> None:
    if not IN_JSON.exists():
        raise FileNotFoundError(f"Input json not found: {IN_JSON}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with IN_JSON.open("r", encoding="utf-8") as f:
        raw: Dict[str, Dict[str, Dict[str, Any]]] = json.load(f)

    raw_train_keys = list(raw.keys())

    # 선택된 train/test 키를 raw json 키로 해석
    selected_train_keys: List[Tuple[DatasetEnum, str]] = []
    for ds in SELECTED:
        k = resolve_key(ds, raw_train_keys)
        if k is None:
            raise KeyError(
                f"Cannot find train key for {ds.name}. "
                f"Available train keys sample: {raw_train_keys[:10]}"
            )
        selected_train_keys.append((ds, k))

    # test 키 후보는 train별로 다를 수 있으니, 우선 전체에서 가능한 키를 모아 해석
    # (보통 모든 train이 동일한 test set을 갖는 구조)
    all_test_keys = set()
    for _, tk in selected_train_keys:
        all_test_keys.update(list(raw[tk].keys()))
    all_test_keys = sorted(all_test_keys)

    selected_test_keys: List[Tuple[DatasetEnum, str]] = []
    for ds in SELECTED:
        k = resolve_key(ds, list(all_test_keys))
        if k is None:
            # 일부 json은 train 쪽에는 있는데 test 쪽엔 없을 수도 있으므로,
            # 이 경우는 스킵하지 말고 에러로 알려서 사용자가 확인하게 하는 편이 안전합니다.
            raise KeyError(
                f"Cannot find test key for {ds.name}. "
                f"Available test keys sample: {list(all_test_keys)[:10]}"
            )
        selected_test_keys.append((ds, k))

    # 필터링 JSON 만들기 (키는 보기 좋게 Enum.name으로 통일)
    filtered: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for ds_train, k_train in selected_train_keys:
        row = raw[k_train]
        out_row: Dict[str, Dict[str, Any]] = {}
        for ds_test, k_test in selected_test_keys:
            if k_test not in row:
                # 없는 경우는 빈 값 처리 (acc/loss/n=None)
                out_row[ds_test.name] = {"loss": None, "acc": None, "n": None}
            else:
                cell = extract_cell(row[k_test])
                out_row[ds_test.name] = {
                    "loss": cell.loss,
                    "acc": cell.acc,
                    "n": cell.n,
                }
        filtered[ds_train.name] = out_row

    out_json_path = OUT_DIR / "cross_test_retest_selected.json"
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    # Heatmap (acc) 만들기
    labels = [ds.name for ds in SELECTED]
    m = len(labels)
    acc_mat = np.full((m, m), np.nan, dtype=np.float32)

    for i, tr in enumerate(labels):
        for j, te in enumerate(labels):
            v = filtered[tr][te].get("acc", None)
            if v is None:
                continue
            try:
                acc_mat[i, j] = float(v)
            except Exception:
                pass

    fig_w = max(10, int(0.9 * m) + 6)
    fig_h = max(8, int(0.7 * m) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(acc_mat, aspect="auto")  # colormap default = viridis

    ax.set_title("Cross-test Accuracy (Selected)")
    ax.set_xlabel("Test dataset")
    ax.set_ylabel("Train dataset")

    ax.set_xticks(np.arange(m))
    ax.set_yticks(np.arange(m))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    # 값 annotate
    for i in range(m):
        for j in range(m):
            if np.isfinite(acc_mat[i, j]):
                ax.text(
                    j, i, f"{acc_mat[i, j]:.3f}", ha="center", va="center", fontsize=9
                )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out_img_path = OUT_DIR / "cross_test_retest_selected.png"
    fig.savefig(out_img_path, dpi=200)
    plt.close(fig)

    print(f"[OK] saved json : {out_json_path}")
    print(f"[OK] saved image: {out_img_path}")


if __name__ == "__main__":
    main()
