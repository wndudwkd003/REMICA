# worker/test_memory.py

import itertools
import json
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from config.config import ChainEnum, Config, DatasetEnum
from enums.out_schema import GPTInferOut
from utils.faiss_utils import (
    pick_topk_with_both_labels_in_order,
    search_similar_by_text,
)
from utils.gpu_utils import make_plan
from utils.metrics_utils import compute_binary_metrics, save_metrics_csv_and_plots
from utils.path_utils import get_memory_root, get_test_root, make_unique_dir
from utils.prompt_builder_debate import build_prompt_test_infer
from utils.time_utils import now_ms
import utils.process_utils as pu

MP_CTX = mp.get_context("spawn")

G_MEMORY_BY_ID = None

_SAFE_CHARS = re.compile(r"[^a-zA-Z0-9._-]+")


def safe_filename(s: str, fallback: str = "noid") -> str:
    s = (s or "").strip()
    if not s:
        return fallback
    s = s.replace("/", "_").replace("\\", "_")
    s = _SAFE_CHARS.sub("_", s)
    if len(s) > 180:
        s = s[:180]
    return s


def tqdm_write(msg: str) -> None:
    tqdm.write(msg)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def init_worker_with_cache(gpu_id: int, config: Config) -> None:
    global G_MEMORY_BY_ID
    pu.init_worker(gpu_id, config)
    G_MEMORY_BY_ID = {}


def ensure_memory_by_id(dataset: DatasetEnum, config: Config) -> dict:
    key = dataset.name
    if key in G_MEMORY_BY_ID:
        return G_MEMORY_BY_ID[key]

    root = get_memory_root(config)
    jsonl_path = root / f"{dataset.name}.jsonl"

    memory_by_id = {}
    if jsonl_path.is_file():
        for rec in iter_jsonl(jsonl_path):
            memory_by_id[str(rec["id"])] = rec

    G_MEMORY_BY_ID[key] = memory_by_id
    return memory_by_id


def worker_run_one(dataset_name: str, row: dict):
    pid = os.getpid()

    sample_id = str(row.get("id", ""))
    text = row.get("text", "")
    label = int(row.get("label", 0))

    rec = {
        "dataset": dataset_name,
        "id": sample_id,
        "text": text,
        "label": label,
        "ok": False,
        "pid": pid,
        "time_ms": 0.0,
        "memories": [],
        "prompt": None,
        "model_out": None,
        "pred": None,
        "pred_used_for_metric": None,
        "error": None,
    }

    try:
        dataset = DatasetEnum[dataset_name]
        index, meta_train = pu.get_index_meta(dataset)
        infer_client = pu.ensure_infer_client()

        use_k = int(pu.G_CONFIG.rag_top_k)

        if use_k <= 0:
            memories = []
        else:
            cand_k = use_k * int(pu.G_CONFIG.rag_cand_k)

            candidates = search_similar_by_text(
                config=pu.G_CONFIG,
                dataset=dataset,
                index=index,
                meta_train=meta_train,
                query_text=text,
                top_k=cand_k,
                query_id=sample_id,
            )

            candidates_full = []

            if pu.G_CONFIG.chain_mode == ChainEnum.NONE:
                for ex in candidates:
                    candidates_full.append(
                        {
                            "id": str(ex["id"]),
                            "text": ex["text"],
                            "label": int(ex["label"]),
                        }
                    )
            else:
                memory_by_id = ensure_memory_by_id(dataset, pu.G_CONFIG)
                for ex in candidates:
                    mid = str(ex["id"])
                    mem_core = memory_by_id.get(mid)
                    if mem_core is None:
                        continue
                    if (
                        not mem_core.get("A1")
                        or not mem_core.get("A2")
                        or not mem_core.get("B1")
                        or not mem_core.get("B2")
                    ):
                        continue
                    candidates_full.append(
                        {
                            "id": mid,
                            "text": mem_core["text"],
                            "label": int(mem_core["label"]),
                            "A1": mem_core["A1"],
                            "A2": mem_core["A2"],
                            "B1": mem_core["B1"],
                            "B2": mem_core["B2"],
                        }
                    )

            memories = pick_topk_with_both_labels_in_order(candidates_full, use_k)

        prompt = build_prompt_test_infer(
            target_text=text,
            dataset=dataset,
            chain_mode=pu.G_CONFIG.chain_mode,
            memories=memories,
        )

        rec["memories"] = memories
        rec["prompt"] = prompt

        t0 = now_ms()
        out_dict = infer_client.call_api(prompt)
        t1 = now_ms()

        rec["time_ms"] = float(t1 - t0)
        rec["model_out"] = out_dict

        infer_out = GPTInferOut.model_validate(out_dict)
        pred = int(infer_out.pred_label)

        rec["ok"] = True
        rec["pred"] = pred
        rec["pred_used_for_metric"] = pred
        rec["error"] = None

        return rec

    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["pred"] = None
        rec["pred_used_for_metric"] = 1 - label
        return rec


def run_test_memory(config: Config) -> None:
    model_root = get_test_root(config)
    eval_dir = make_unique_dir(model_root / "eval")
    tqdm_write(f"[run_test_memory] eval_dir = {eval_dir}")

    pred_root = eval_dir / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    plan = make_plan(config)
    executors = []
    for gid, n_workers in plan:
        executors.append(
            ProcessPoolExecutor(
                mp_context=MP_CTX,
                max_workers=n_workers,
                initializer=init_worker_with_cache,
                initargs=(gid, config),
            )
        )

    rr = itertools.cycle(executors)

    try:
        for dataset, _bs in config.dataset_order:
            tqdm_write(f"\n[test_memory] dataset = {dataset.name}")

            test_path = Path(config.datasets_dir) / dataset.name / "test.jsonl"
            if not test_path.is_file():
                tqdm_write(f"  - test.jsonl not found, skip: {test_path}")
                continue

            if config.chain_mode != ChainEnum.NONE:
                mem_path = get_memory_root(config) / f"{dataset.name}.jsonl"
                if not mem_path.is_file():
                    tqdm_write(f"  - memory jsonl not found, skip: {mem_path}")
                    continue

            futures = []
            future_to_input = {}

            for row in iter_jsonl(test_path):
                ex = next(rr)
                fut = ex.submit(worker_run_one, dataset.name, row)
                futures.append(fut)
                future_to_input[fut] = {
                    "dataset": dataset.name,
                    "id": str(row.get("id", "")),
                    "text": row.get("text", ""),
                    "label": int(row.get("label", 0)),
                }

            if not futures:
                tqdm_write(f"  - no test samples: {test_path}")
                continue

            ok_count = 0
            fail_count = 0

            pbar = tqdm(
                desc=f"test_memory {dataset.name}[test]",
                total=len(futures),
                unit="sample",
                dynamic_ncols=True,
            )
            pbar.set_postfix(ok=ok_count, fail=fail_count)

            y_true = []
            y_pred = []
            times_ms = []

            for fut in as_completed(futures):
                base = future_to_input[fut]
                true_y = int(base["label"])

                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {
                        "dataset": base["dataset"],
                        "id": base["id"],
                        "text": base["text"],
                        "label": true_y,
                        "ok": False,
                        "pid": None,
                        "time_ms": 0.0,
                        "memories": [],
                        "prompt": None,
                        "model_out": None,
                        "pred": None,
                        "pred_used_for_metric": 1 - true_y,
                        "error": f"{type(e).__name__}: {e}",
                    }

                ds_dir = pred_root / base["dataset"]
                ds_dir.mkdir(parents=True, exist_ok=True)

                sid = safe_filename(str(rec.get("id", "")), fallback="noid")
                out_path = ds_dir / f"{sid}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(rec, f, ensure_ascii=False, indent=2)

                if rec.get("ok", False):
                    ok_count += 1
                else:
                    fail_count += 1

                y_true.append(true_y)
                y_pred.append(int(rec["pred_used_for_metric"]))
                times_ms.append(float(rec.get("time_ms", 0.0) or 0.0))

                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)

            pbar.close()

            if y_true:
                metrics = compute_binary_metrics(y_true, y_pred)
                metrics["n_ok"] = int(ok_count)
                metrics["n_fail"] = int(fail_count)
                metrics["n_total"] = int(ok_count + fail_count)
                metrics["fail_rate"] = (
                    float(fail_count / (ok_count + fail_count))
                    if (ok_count + fail_count) > 0
                    else 0.0
                )

                save_metrics_csv_and_plots(
                    config=config,
                    dataset=dataset,
                    metrics=metrics,
                    out_dir=eval_dir,
                )

                time_info = {
                    "n_samples": len(times_ms),
                    "median_ms": float(np.median(times_ms)) if times_ms else 0.0,
                    "mean_ms": float(np.mean(times_ms)) if times_ms else 0.0,
                }
                out_time = eval_dir / f"{dataset.name}_time.json"
                with out_time.open("w", encoding="utf-8") as f:
                    json.dump(time_info, f, ensure_ascii=False, indent=2)

                tqdm_write(f"  - metrics & time saved: {dataset.name}")

    finally:
        for ex in executors:
            ex.shutdown(wait=True, cancel_futures=False)
