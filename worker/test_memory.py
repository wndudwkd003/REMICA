# worker/test_memory.py


import itertools
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from config.config import ChainEnum, Config, DatasetEnum
from enums.out_schema import GPTInferOut
from utils.faiss_utils import (
    search_similar_by_text,
    pick_topk_with_both_labels_in_order,
)
from utils.gpu_utils import make_plan
from utils.metrics_utils import compute_binary_metrics, save_metrics_csv_and_plots
from utils.path_utils import get_memory_root, get_test_root
from utils.prompt_builder_debate import build_prompt_test_infer
from utils.time_utils import now_ms
import utils.process_utils as pu

MP_CTX = mp.get_context("spawn")

G_MEMORY_BY_ID = None


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


def make_unique_dir(base_dir: Path) -> Path:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=False)
        return base_dir

    i = 1
    while True:
        cand = Path(str(base_dir) + f"_{i}")
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand
        i += 1


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

    try:
        dataset = DatasetEnum[dataset_name]
        index, meta_train = pu.get_index_meta(dataset)
        infer_client = pu.ensure_infer_client()

        sample_id = str(row["id"])
        text = row["text"]
        label = int(row["label"])

        cand_k = pu.G_CONFIG.rag_top_k * 30
        use_k = pu.G_CONFIG.rag_top_k

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

        if len(memories) == 0:
            raise ValueError("No valid memories found for inference.")

        prompt = build_prompt_test_infer(
            target_text=text,
            dataset=dataset,
            chain_mode=pu.G_CONFIG.chain_mode,
            memories=memories,
        )

        t0 = now_ms()
        out_dict = infer_client.call_api(prompt)
        t1 = now_ms()

        infer_out = GPTInferOut.model_validate(out_dict)
        pred = int(infer_out.pred_label)

        return True, label, pred, float(t1 - t0), None, pid

    except Exception as e:
        label = int(row.get("label", 0))
        return False, label, None, 0.0, f"{type(e).__name__}: {e}", pid


def run_test_memory(config: Config) -> None:
    model_root = get_test_root(config)
    eval_dir = make_unique_dir(model_root / "eval")
    tqdm_write(f"[run_test_memory] eval_dir = {eval_dir}")

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
            for row in iter_jsonl(test_path):
                ex = next(rr)
                futures.append(ex.submit(worker_run_one, dataset.name, row))

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
                try:
                    ok, y, p, time_ms, err, pid = fut.result()
                except Exception:
                    fail_count += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=ok_count, fail=fail_count)
                    continue

                if not ok:
                    fail_count += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=ok_count, fail=fail_count)
                    continue

                y_true.append(int(y))
                y_pred.append(int(p))
                times_ms.append(float(time_ms))

                ok_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)

            pbar.close()

            if y_true:
                metrics = compute_binary_metrics(y_true, y_pred)
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
