# worker/online_mode.py
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
from config.config import ChainEnum, Config, DatasetEnum
from utils.gpu_utils import make_plan
from utils.prompt_builder_debate import build_prompt_chain_step
import utils.process_utils as pu
from utils.time_utils import now_ms
from utils.path_utils import get_online_root, make_unique_dir
from enums.out_schema import GPTInferOut
from utils.prompt_builder_debate import build_prompt_chain_step, build_prompt_test_infer
from utils.faiss_utils import (
    pick_topk_with_both_labels_in_order,
    search_similar_by_text,
)
from utils.metrics_utils import compute_binary_metrics, save_metrics_csv_and_plots

MP_CTX = mp.get_context("spawn")


def tqdm_write(msg: str) -> None:
    tqdm.write(msg)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()


def worker_online_one(dataset_name: str, row: dict):
    pid = os.getpid()
    sample_id = str(row["id"])

    try:
        dataset = DatasetEnum[dataset_name]
        index, meta_train = pu.get_index_meta(dataset)
        infer_client = pu.ensure_infer_client()


        text = row["text"]
        label = row["label"]

        use_k = int(pu.G_CONFIG.rag_top_k)

        candidates_full: list[dict] = []


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

            for ex in candidates:
                candidates_full.append(
                    {
                        "id": ex["id"],
                        "text": ex["text"],
                        "label": ex["label"],
                    }
                )

            memories = pick_topk_with_both_labels_in_order(candidates_full, use_k)


        out_A1 = None
        out_B1 = None
        out_A2 = None
        out_B2 = None

        steps = (1, 2, 3, 4)
        trace = {}

        # 온라인 평가에서는 정답 힌트 금지
        actual_label_intervention = False

        t0 = now_ms()

        # -------------------------
        # A1/B1/A2/B2 체인 수행
        # -------------------------
        for step_index in steps:
            step_name, prompt = build_prompt_chain_step(
                step_index=step_index,
                target_text=text,
                dataset=dataset,
                similar_examples=memories,
                out_A1=out_A1,
                out_B1=out_B1,
                out_A2=out_A2,
                out_B2=out_B2,
                actual_label_intervention=actual_label_intervention,
                actual_label=None,
                chain_mode=pu.G_CONFIG.chain_mode,
            )

            client_key = "A" if step_name.startswith("A") else "B"
            out_raw = pu.G_CLIENTS[client_key].call_api(prompt)
            out_dict = pu.validate_step_output(step_name, out_raw)

            trace[step_name] = {
                "output": out_dict,
                "prompt": prompt,
            }

            if step_name == "A1":
                out_A1 = out_dict
            elif step_name == "B1":
                out_B1 = out_dict
            elif step_name == "A2":
                out_A2 = out_dict
            elif step_name == "B2":
                out_B2 = out_dict

        # -------------------------
        # 최종 INFER 단계 (test_memory와 동일)
        # -------------------------
        # build_prompt_test_infer는 "memories" 리스트를 받으므로,
        # 온라인에서는 체인 trace(자기 자신 1개)를 memory 1개로 넣어줍니다.
        memories = [
            {
                "text": text,
                "label": "No correct answer is provided.", # 온라인 평가에서는 정답 힌트 금지
                "A1": {
                    "prompt": trace["A1"]["prompt"],
                    "output": trace["A1"]["output"],
                },
                "B1": {
                    "prompt": trace["B1"]["prompt"],
                    "output": trace["B1"]["output"],
                },
                "A2": {
                    "prompt": trace["A2"]["prompt"],
                    "output": trace["A2"]["output"],
                },
                "B2": {
                    "prompt": trace["B2"]["prompt"],
                    "output": trace["B2"]["output"],
                },
            }
        ]

        infer_prompt = build_prompt_test_infer(
            target_text=text,
            dataset=dataset,
            chain_mode=pu.G_CONFIG.chain_mode,
            memories=memories,
        )

        infer_client = pu.ensure_infer_client()
        infer_raw = infer_client.call_api(infer_prompt)
        infer_out = GPTInferOut.model_validate(infer_raw).model_dump()
        pred_label = int(infer_out["pred_label"])

        trace["INFER"] = {
            "output": infer_out,
            "prompt": infer_prompt,
        }

        t1 = now_ms()

        rec = {
            "id": sample_id,
            "text": text,
            "label": label,
            "pred_label": pred_label,
            "chain_mode": pu.G_CONFIG.chain_mode.value,
            "actual_label_intervention": actual_label_intervention,
            "trace": trace,
        }

        return True, rec, float(t1 - t0), None, pid

    except Exception as e:
        return False, {"id": sample_id}, 0.0, f"{type(e).__name__}: {e}", pid


def run_online_mode(config: Config) -> None:
    if config.chain_mode == ChainEnum.NONE:
        raise ValueError(
            "online_mode는 chain_mode != NONE 이어야 합니다. (DEBATE/EXPERT만 지원)"
        )

    run_root = get_online_root(config)
    run_root = make_unique_dir(run_root / "eval")
    tqdm_write(f"[run_online_mode] run_root = {run_root}")

    plan = make_plan(config)
    executors = []
    for gid, n_workers in plan:
        executors.append(
            ProcessPoolExecutor(
                mp_context=MP_CTX,
                max_workers=n_workers,
                initializer=pu.init_worker,
                initargs=(gid, config),
            )
        )

    rr = itertools.cycle(executors)

    try:
        for dataset, _bs in config.dataset_order:

            base = Path(config.datasets_dir) / dataset.name
            split_path = base / "test.jsonl"
            tqdm_write(f"\n[online] dataset = {dataset.name} | split = {split_path}")

            out_path = run_root / "predictions" / f"{dataset.name}.pred.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists():
                out_path.unlink()

            futures = []
            future_to_true = {}

            for row in iter_jsonl(split_path):
                ex = next(rr)
                fut = ex.submit(worker_online_one, dataset.name, row)
                futures.append(fut)
                future_to_true[fut] = row.get("label", None)

            if not futures:
                tqdm_write(f"  - no samples: {split_path}")
                continue

            ok_count = 0
            fail_count = 0

            y_true = []
            y_pred = []
            times_ms = []

            pbar = tqdm(
                desc=f"online {dataset.name}",
                total=len(futures),
                unit="sample",
                dynamic_ncols=True,
            )
            pbar.set_postfix(ok=ok_count, fail=fail_count)

            for fut in as_completed(futures):
                true_y = future_to_true.get(fut, None)

                try:
                    ok, rec, time_ms, err, pid = fut.result()
                except Exception:
                    # ---- 실패 = 오답 처리 (label이 있는 경우만) ----
                    fail_count += 1
                    if true_y is not None:
                        true_y = int(true_y)
                        y_true.append(true_y)
                        y_pred.append(1 - true_y)
                    times_ms.append(0.0)

                    pbar.update(1)
                    pbar.set_postfix(ok=ok_count, fail=fail_count)
                    continue

                if not ok:
                    # ---- 실패 = 오답 처리 (label이 있는 경우만) ----
                    fail_count += 1
                    if true_y is not None:
                        true_y = int(true_y)
                        y_true.append(true_y)
                        y_pred.append(1 - true_y)
                    times_ms.append(float(time_ms) if time_ms else 0.0)

                    tqdm_write(
                        f"[online][FAIL] {dataset.name} | SID={rec.get('id')} | {err} | pid={pid}"
                    )
                    pbar.update(1)
                    pbar.set_postfix(ok=ok_count, fail=fail_count)
                    continue

                # ---- 성공 케이스 ----
                ok_count += 1
                append_jsonl(out_path, rec)

                y_true.append(int(rec["label"]))
                y_pred.append(int(rec["pred_label"]))

                times_ms.append(float(time_ms) if time_ms else 0.0)

                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)

            pbar.close()
            tqdm_write(f"[online] saved: {out_path} | ok={ok_count}, fail={fail_count}")

            # ----------------------------
            # metrics 저장: test_memory와 동일
            # ----------------------------
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
                    out_dir=run_root,
                )

                # ----------------------------
                # time.json 저장: test_memory와 동일
                # ----------------------------
                time_info = {
                    "n_samples": len(times_ms),
                    "median_ms": float(np.median(times_ms)) if times_ms else 0.0,
                    "mean_ms": float(np.mean(times_ms)) if times_ms else 0.0,
                }
                out_time = run_root / f"{dataset.name}_time.json"
                with out_time.open("w", encoding="utf-8") as f:
                    json.dump(time_info, f, ensure_ascii=False, indent=2)

                tqdm_write(f"  - metrics & time saved: {dataset.name}")

    finally:
        for ex in executors:
            ex.shutdown(wait=True, cancel_futures=False)
