# worker/gpt_infer.py

import json
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from config.config import Config, DatasetEnum
from utils.data_utils import JsonlDataset
from utils.rem2_retriever import Rem2Retriever, Rem2AugDataset

from openai import OpenAI


# -----------------------------
# JSON utils
# -----------------------------


def _json_default(o):
    from enum import Enum

    if isinstance(o, Enum):
        return o.name
    return str(o)


def _write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")


def _safe_int01(x: Any) -> int:
    try:
        v = int(x)
        return 1 if v != 0 else 0
    except Exception:
        return 0


def make_run_dir_gpt(config: Config) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # gpt 모델명을 폴더명에 안전하게 넣기
    gpt_tag = str(config.gpt_model)
    gpt_tag = gpt_tag.replace("/", "_").replace(":", "_").replace(" ", "_")

    run_name = f"{ts}_GPT_{gpt_tag}"
    run_dir = Path(config.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "save").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    cfg = asdict(config) if is_dataclass(config) else dict(config.__dict__)
    cfg["dataset_order"] = [(ds.name, bs) for ds, bs in config.dataset_order]
    cfg["_run_mode"] = "GPT_INFER"
    cfg["_created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _write_json(run_dir / "config.json", cfg)
    return run_dir


def dataset_paths(config: Config, dataset: DatasetEnum):
    base = Path(config.datasets_dir) / dataset.name
    return {
        "train": base / "train.jsonl",
        "valid": base / "valid.jsonl",
        "test": base / "test.jsonl",
    }


# -----------------------------
# 결과 grid plot (1 x N)
# -----------------------------
def plot_1xN_grid(
    values: List[float],
    col_names: List[str],
    out_path: Path,
    title: str,
    fmt: str = ".3f",
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(values, dtype=np.float32).reshape(1, -1)
    fig = plt.figure(figsize=(max(8, len(col_names) * 1.1), 2.2))
    ax = plt.gca()
    ax.imshow(data, aspect="auto")

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


# -----------------------------
# GPT client (판별 중심) + schema 강제
# -----------------------------
class GPTInferOut(BaseModel):
    pred_label: int = Field(..., description="0 or 1")
    rationale: str = Field(
        ..., description="Optional short rationale; may be empty string"
    )


def _build_prompt(text: str) -> Tuple[str, str]:
    system = (
        "You are a toxicity/safety text classifier.\n"
        "Return only a JSON object that matches the provided JSON Schema.\n"
        "No extra text.\n"
        "\n"
        "Label definition:\n"
        "- pred_label=0: normal / harmless\n"
        "- pred_label=1: inappropriate / toxic (includes hate/offensive)\n"
    )

    user = (
        "Classify the text below according to the label definition.\n"
        "The classification (pred_label) is the most important.\n"
        "Text:\n"
        f"{text}\n"
    )
    return system, user


def _response_format_schema() -> Dict[str, Any]:
    # Responses API 구조적 출력(JSON Schema) 강제
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "toxicity_classifier",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "pred_label": {"type": "integer", "enum": [0, 1]},
                    "rationale": {"type": "string"},
                },
                "required": ["pred_label", "rationale"],
                "additionalProperties": False,
            },
        },
    }


def _pydantic_validate(js: Dict[str, Any]) -> GPTInferOut:
    # pydantic v2 우선, 없으면 v1 방식으로 처리 (getattr 사용 없음)
    try:
        return GPTInferOut.model_validate(js)  # pydantic v2
    except AttributeError:
        return GPTInferOut.parse_obj(js)  # pydantic v1


def gpt_predict_one(
    client: OpenAI,
    config: Config,
    text: str,
) -> Dict[str, Any]:
    system, user = _build_prompt(text)

    last_err: Exception | None = None

    for attempt in range(int(config.max_retries)):
        try:
            tqdm.write(
                "\n" + "=" * 20 + " [입력 텍스트: SYSTEM] " + "=" * 20 + "\n" + system
            )
            tqdm.write(
                "\n" + "=" * 20 + " [입력 텍스트: USER] " + "=" * 22 + "\n" + user
            )

            resp = client.responses.parse(
                model=config.gpt_model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=config.gpt_temperature,
                max_output_tokens=config.gpt_max_output_tokens,
                text_format=GPTInferOut,
                top_p=config.gpt_top_p,
            )

            # 1) 거절(refusal) 처리 (가이드에 나오는 케이스)
            # SDK/모델에 따라 refusal 위치가 다를 수 있는데,
            # responses.parse를 쓰면 보통 output_text에는 refusal 텍스트가 들어갈 수 있습니다.
            out_text = resp.output_text.strip()
            tqdm.write(
                "\n" + "=" * 20 + " [출력 텍스트] " + "=" * 31 + "\n" + out_text + "\n"
            )

            # 2) 파싱된 객체 (Pydantic)
            parsed = resp.output_parsed
            if parsed is None:
                # parsed가 비면 out_text만 보고 재시도 or fallback
                raise ValueError("output_parsed is None")

            pred_label = _safe_int01(parsed.pred_label)
            rationale = (parsed.rationale or "").strip()
            return {"pred_label": pred_label, "rationale": rationale}

        except Exception as e:
            last_err = e
            tqdm.write(
                "\n"
                + "!" * 20
                + f" [API_ERROR] attempt={attempt+1}/{int(config.max_retries)} "
                + "!" * 20
                + f"\n{type(e).__name__}: {str(e)}\n"
            )
            time.sleep(0.5 + 0.2 * attempt)

    tqdm.write(
        "\n"
        + "!" * 20
        + " [API_ERROR] FINAL FAILED "
        + "!" * 20
        + f"\n{type(last_err).__name__ if last_err else 'Unknown'}: {str(last_err) if last_err else ''}\n"
    )
    return {"pred_label": 0, "rationale": f"[gpt_error] {str(last_err)[:200]}"}


# -----------------------------
# main entry
# -----------------------------
def run_gpt_infer(config: Config) -> str:
    run_dir = make_run_dir_gpt(config)

    rem2 = None
    if config.use_rem2_aug:
        device_str = "cuda" if str(config.emb_device).lower() == "cuda" else "cpu"
        rem2 = Rem2Retriever(config, device=device_str)

    client = OpenAI()

    cross_rows: List[Dict[str, Any]] = []

    outer = tqdm(config.dataset_order, desc="gpt-infer-datasets")
    try:
        for ds, _ in outer:
            ds: DatasetEnum
            ds_name = ds.name

            save_dir = run_dir / "save" / ds_name
            save_dir.mkdir(parents=True, exist_ok=True)

            paths = dataset_paths(config, ds)
            test_path = paths["test"]
            assert test_path.exists(), f"test.jsonl not found: {test_path}"

            base = JsonlDataset(str(test_path), meta_to_text=config.meta_to_text)
            ds_obj = Rem2AugDataset(base, rem2) if rem2 is not None else base

            preds: List[dict] = []
            y_true: List[int] = []
            y_pred: List[int] = []

            pbar = tqdm(range(len(ds_obj)), desc=f"gpt-test({ds_name})", leave=False)
            for i in pbar:
                sid, text, label, meta = ds_obj[i]
                y = _safe_int01(label)

                out = gpt_predict_one(client, config, text)
                yp = _safe_int01(out["pred_label"])
                rat = (out.get("rationale") or "").strip()

                preds.append(
                    {
                        "sid": sid,
                        "y_true": y,
                        "y_pred": yp,
                        "rationale": rat,
                    }
                )
                y_true.append(y)
                y_pred.append(yp)

            acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
            macro_f1 = (
                float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0
            )

            _write_jsonl(save_dir / "pred_test.jsonl", preds)
            _write_json(
                save_dir / "meta.json",
                {
                    "mode": "GPT_INFER",
                    "dataset": ds_name,
                    "split": "test",
                    "n": len(y_true),
                    "acc": acc,
                    "macro_f1": macro_f1,
                    "gpt_model": str(config.gpt_model),
                    "gpt_temperature": float(config.gpt_temperature),
                    "use_rem2_aug": bool(config.use_rem2_aug),
                    "rem2_top_k": int(config.rem2_top_k),
                    "rem2_min_reliability": float(config.rem2_min_reliability),
                    "rem2_only_correct": bool(config.rem2_only_correct),
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )

            cross_rows.append(
                {
                    "test_dataset": ds_name,
                    "acc": acc,
                    "macro_f1": macro_f1,
                    "n": float(len(y_true)),
                }
            )
            outer.set_postfix(ds=ds_name, acc=f"{acc:.3f}", f1=f"{macro_f1:.3f}")

    finally:
        if rem2 is not None:
            rem2.close()

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(cross_rows)
    df.to_csv(results_dir / "cross_test.csv", index=False)
    _write_json(
        results_dir / "cross_test.json", {"mode": "GPT_INFER", "rows": cross_rows}
    )

    order_names = [ds.name for ds, _ in config.dataset_order]
    df2 = df.set_index("test_dataset").reindex(order_names).reset_index()

    acc_vals = [float(x) if x == x else 0.0 for x in df2["acc"].tolist()]
    f1_vals = [float(x) if x == x else 0.0 for x in df2["macro_f1"].tolist()]

    plot_1xN_grid(
        values=acc_vals,
        col_names=order_names,
        out_path=results_dir / "cross_test_grid_acc.png",
        title="GPT Test Accuracy (by dataset)",
        fmt=".3f",
    )
    plot_1xN_grid(
        values=f1_vals,
        col_names=order_names,
        out_path=results_dir / "cross_test_grid_f1.png",
        title="GPT Test Macro-F1 (by dataset)",
        fmt=".3f",
    )

    print(f"\n[DONE] gpt_infer run_dir={run_dir}")
    return str(run_dir)
