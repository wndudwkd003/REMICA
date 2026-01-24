# worker/gpt_infer.py

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from config.config import Config, DatasetEnum
from utils.client_utils import GPTClient, GPTInferOut
from utils.cuda_utils import get_device
from utils.data_utils import JsonlDataset
from utils.dir_utils import make_run_dir_gpt
from utils.prompt_utils import GPT_INFER_SYSTEM
from utils.rem2_retriever import Rem2ExampleAugDataset, Rem2Retriever
from utils.viz_utils import plot_1xN_grid

# =========================================================
# 공통 유틸
# =========================================================

def _json_default(o):
    from enum import Enum
    if isinstance(o, Enum):
        return o.name
    return str(o)


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def dataset_paths(config: Config, dataset: DatasetEnum):
    base = Path(config.datasets_dir) / dataset.name
    return {
        "train": base / "train.jsonl",
        "valid": base / "valid.jsonl",
        "test": base / "test.jsonl",
    }


def build_gpt_infer_prompt(content: str) -> str:
    """
    content:
      - mode='plain'  : 그냥 원문
      - mode='rem1'   : TARGET_TEXT + 유사샘플 REM1
      - mode='rem12'  : TARGET_TEXT + 유사샘플 REM1+REM2
    전부 하나의 user 프롬프트로 합쳐서 GPTClient에 넣습니다.
    """
    return f"""{GPT_INFER_SYSTEM}

TARGET_TEXT:
{content}
"""


# =========================================================
# mode별 Dataset 래핑
# =========================================================

# build_dataset_object 쪽

def build_dataset_object(
    config: Config,
    ds: DatasetEnum,
    retriever: Rem2Retriever | None,
    mode: str,
):
    paths = dataset_paths(config, ds)
    test_path = paths["test"]
    assert test_path.exists(), f"test.jsonl not found: {test_path}"

    base = JsonlDataset(str(test_path), meta_to_text=config.meta_to_text)

    if mode == "plain":
        return base

    assert retriever is not None
    top_k = config.rem2_top_k

    # 여기서도 레지스트리 리스트 하나만 넘김
    pairs = [(ds, base)]
    return Rem2ExampleAugDataset(
        datasets=pairs,
        retriever=retriever,
        top_k=top_k,
        mode=mode,
    )



def eval_dataset(
    client: GPTClient[GPTInferOut],
    config: Config,
    ds: DatasetEnum,
    retriever: Rem2Retriever | None,
    mode: str,
    save_dir: Path,
) -> Dict[str, Any]:
    ds_name = ds.name
    save_dir.mkdir(parents=True, exist_ok=True)

    ds_obj = build_dataset_object(config, ds, retriever, mode)

    preds: List[dict] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    pbar = tqdm(range(len(ds_obj)), desc=f"gpt-test({ds_name})[{mode}]", leave=False)
    for i in pbar:
        sid, text, label, meta = ds_obj[i]

        # ----- 프롬프트 구성 + 로깅 -----
        prompt = build_gpt_infer_prompt(text)

        tqdm.write(
            "\n"
            + "=" * 20
            + f" [GPT_INFER INPUT] ds={ds_name} mode={mode} sid={sid} "
            + "=" * 20
        )
        tqdm.write(prompt)

        # ----- GPT 호출 -----
        out = client.call_api(prompt)

        tqdm.write(
            "\n"
            + "=" * 20
            + f" [GPT_INFER OUTPUT] ds={ds_name} mode={mode} sid={sid} "
            + "=" * 20
        )
        tqdm.write(json.dumps(out, ensure_ascii=False))

        pred_label = int(out["pred_label"])
        rationale = (out.get("rationale") or "").strip()

        # jsonl에 "답변(pred_label) + 이유(rationale)" 저장
        preds.append(
            {
                "sid": sid,
                "y_true": int(label),
                "y_pred": pred_label,
                "rationale": rationale,
            }
        )
        y_true.append(int(label))
        y_pred.append(pred_label)

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    macro_f1 = (
        float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0
    )

    # 샘플별 pred + rationale
    write_jsonl(save_dir / f"pred_test_{mode}.jsonl", preds)

    # 데이터셋 전체 메타
    write_json(
        save_dir / f"meta_{mode}.json",
        {
            "mode": f"GPT_INFER_{mode}",
            "dataset": ds_name,
            "split": "test",
            "n": len(y_true),
            "acc": acc,
            "macro_f1": macro_f1,
            "gpt_model": str(config.gpt_model),
            "gpt_temperature": float(config.gpt_temperature),
            "gpt_infer_mode": mode,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    return {
        "test_dataset": ds_name,
        "acc": acc,
        "macro_f1": macro_f1,
        "n": float(len(y_true)),
    }


# =========================================================
# main entry
# =========================================================

def run_gpt_infer(config: Config) -> str:
    run_dir = make_run_dir_gpt(config)
    mode = config.rem_mode  # "plain" | "rem1" | "rem12"

    gpt_client = GPTClient[GPTInferOut](
        model=config.gpt_model,
        max_output_tokens=config.gpt_max_output_tokens,
        schema=GPTInferOut,
        max_retries=config.max_retries,
        retry_sleep=0.5,
        temperature=config.gpt_temperature,
        top_p=config.gpt_top_p,
    )

    retriever: Rem2Retriever | None = None
    if mode in ("rem1", "rem12"):
        retriever = Rem2Retriever(config, device=get_device())

    cross_rows: List[Dict[str, Any]] = []

    outer = tqdm(config.dataset_order, desc=f"gpt-infer-datasets[{mode}]")
    try:
        for ds, _ in outer:
            save_dir = run_dir / "save" / ds.name

            row = eval_dataset(
                client=gpt_client,
                config=config,
                ds=ds,
                retriever=retriever,
                mode=mode,
                save_dir=save_dir,
            )
            cross_rows.append(row)
            outer.set_postfix(
                ds=ds.name, acc=f"{row['acc']:.3f}", f1=f"{row['macro_f1']:.3f}"
            )
    finally:
        if retriever is not None:
            retriever.close()

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(cross_rows)
    df.to_csv(results_dir / f"cross_test_{mode}.csv", index=False)
    write_json(
        results_dir / f"cross_test_{mode}.json",
        {"mode": f"GPT_INFER_{mode}", "rows": cross_rows},
    )

    order_names = [ds.name for ds, _ in config.dataset_order]
    df2 = df.set_index("test_dataset").reindex(order_names).reset_index()

    acc_vals = [float(x) if x == x else 0.0 for x in df2["acc"].tolist()]
    f1_vals = [float(x) if x == x else 0.0 for x in df2["macro_f1"].tolist()]

    plot_1xN_grid(
        values=acc_vals,
        col_names=order_names,
        out_path=results_dir / f"cross_test_grid_acc_{mode}.png",
        title=f"GPT Test Accuracy (by dataset) [{mode}]",
        fmt=".3f",
    )
    plot_1xN_grid(
        values=f1_vals,
        col_names=order_names,
        out_path=results_dir / f"cross_test_grid_f1_{mode}.png",
        title=f"GPT Test Macro-F1 (by dataset) [{mode}]",
        fmt=".3f",
    )

    print(f"\n[DONE] gpt_infer run_dir={run_dir}  mode={mode}")
    return str(run_dir)
