# # worker/debate_direct.py

# from __future__ import annotations

# import itertools
# import json
# import multiprocessing as mp
# import os
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from pathlib import Path
# from typing import Any, Dict, Tuple

# import numpy as np
# import torch
# from tqdm.auto import tqdm
# from utils.data_utils import load_jsonl
# from config.config import Config, TARGET_DATASETS, DatasetEnum
# from enums.debate_schema import AgentAOut, AgentBOut, AgentCOut, AgentDOut
# from enums.out_schema import GPTInferOut
# from utils.faiss_utils import load_faiss_index_and_meta, search_similar_by_text
# from utils.gpu_utils import make_plan
# from utils.llm_builder import build_agent_client
# from utils.metrics_utils import compute_binary_metrics, save_metrics_csv_and_plots
# from utils.prompt_builder_debate import (
#     build_prompt_debate_final_infer,
#     build_prompt_agent_A,
#     build_prompt_agent_B,
#     build_prompt_agent_C_infer,
#     build_prompt_agent_D_infer,
#     build_prompt_direct_nondebate,
# )
# from utils.time_utils import now_ms

# MP_CTX = mp.get_context("spawn")

# # ----------------------------
# # per-process globals (cache)
# # ----------------------------
# G_GPU_ID: int | None = None
# G_CONFIG: Config | None = None
# G_CLIENTS: Dict[str, Any] | None = None
# G_INDEX_CACHE: Dict[str, Any] | None = None


# def tqdm_write(msg: str) -> None:
#     tqdm.write(msg)


# def init_worker(gpu_id: int, config: Config) -> None:
#     """
#     worker process 시작 시 1회:
#     - GPU 고정
#     - config 저장
#     - LLM clients 생성
#     - FAISS index/meta 캐시 dict 준비
#     """
#     global G_GPU_ID, G_CONFIG, G_CLIENTS, G_INDEX_CACHE

#     G_GPU_ID = int(gpu_id)
#     G_CONFIG = config

#     torch.cuda.set_device(G_GPU_ID)
#     # RAG/임베딩에서 rag_device를 쓰는 구조면 고정
#     G_CONFIG.rag_device = f"cuda:{G_GPU_ID}"

#     # process당 1회 생성 후 재사용
#     G_CLIENTS = {
#         "A": build_agent_client(G_CONFIG, AgentAOut, "A"),
#         "B": build_agent_client(G_CONFIG, AgentBOut, "B"),
#         "C": build_agent_client(G_CONFIG, AgentCOut, "C"),
#         "D": build_agent_client(G_CONFIG, AgentDOut, "D"),
#         # FINAL judge (A/B/C/D outputs 합쳐서 최종 pred_label)
#         "FINAL": build_agent_client(G_CONFIG, GPTInferOut, "INFER"),
#         # nothing-direct 단일패스
#         "DIRECT": build_agent_client(G_CONFIG, GPTInferOut, "INFER"),
#     }

#     G_INDEX_CACHE = {}


# def get_index_meta(dataset: DatasetEnum):
#     """
#     worker process 내에서 dataset별 index/meta_train 1회만 로드.
#     """
#     global G_INDEX_CACHE
#     assert G_INDEX_CACHE is not None
#     key = dataset.name
#     if key in G_INDEX_CACHE:
#         return G_INDEX_CACHE[key]

#     index, meta_train = load_faiss_index_and_meta(
#         config=G_CONFIG,
#         dataset=dataset,
#         split="train",
#     )
#     G_INDEX_CACHE[key] = (index, meta_train)
#     return index, meta_train


# def _validate_row(row: dict) -> Tuple[bool, str, int, str]:
#     text = row.get("text")
#     label = row.get("label")

#     if text is None or label is None:
#         return False, "", 0, "missing text/label"
#     if not isinstance(text, str) or not text.strip():
#         return False, "", 0, "empty text"
#     try:
#         gold = int(label)
#     except Exception:
#         return False, "", 0, "label not int-castable"

#     return True, text, gold, ""


# # =========================================================
# # workers
# # =========================================================
# def worker_debate_direct_one(dataset_name: str, row: dict):
#     """
#     A -> B -> C -> D 수행 후,
#     A/B/C/D outputs를 합쳐 FINAL(INFER)을 1회 더 호출해서 pred_label 결정.
#     """
#     pid = os.getpid()
#     try:
#         dataset = DatasetEnum[dataset_name]
#         ok, target_text, gold, verr = _validate_row(row)
#         if not ok:
#             return False, gold, None, 0.0, f"ValueError: {verr}", pid

#         index, meta_train = get_index_meta(dataset)

#         similar_examples = search_similar_by_text(
#             config=G_CONFIG,
#             dataset=dataset,
#             index=index,
#             meta_train=meta_train,
#             query_text=target_text,
#             top_k=G_CONFIG.rag_top_k,
#         )

#         t0 = now_ms()

#         # A
#         prompt_A = build_prompt_agent_A(
#             target_text=target_text,
#             dataset=dataset,
#             similar_examples=similar_examples,
#         )
#         out_A_dict = G_CLIENTS["A"].call_api(prompt_A)
#         out_A = AgentAOut.model_validate(out_A_dict)

#         # B
#         prompt_B = build_prompt_agent_B(
#             target_text=target_text,
#             dataset=dataset,
#             agent_A_out=out_A,
#         )
#         out_B_dict = G_CLIENTS["B"].call_api(prompt_B)
#         out_B = AgentBOut.model_validate(out_B_dict)

#         # C (gold-free)
#         prompt_C = build_prompt_agent_C_infer(
#             target_text=target_text,
#             dataset=dataset,
#             agent_A_out=out_A,
#             agent_B_out=out_B,
#         )
#         out_C_dict = G_CLIENTS["C"].call_api(prompt_C)
#         out_C = AgentCOut.model_validate(out_C_dict)

#         # D (gold-free)
#         prompt_D = build_prompt_agent_D_infer(
#             target_text=target_text,
#             dataset=dataset,
#             agent_A_out=out_A,
#             agent_B_out=out_B,
#             agent_C_out=out_C,
#         )
#         out_D_dict = G_CLIENTS["D"].call_api(prompt_D)
#         out_D = AgentDOut.model_validate(out_D_dict)

#         # FINAL(INFER)
#         prompt_final = build_prompt_debate_final_infer(
#             target_text=target_text,
#             dataset=dataset,
#             out_A=out_A.model_dump(),
#             out_B=out_B.model_dump(),
#             out_C=out_C.model_dump(),
#             out_D=out_D.model_dump(),
#         )
#         out_final_dict = G_CLIENTS["FINAL"].call_api(prompt_final)
#         out_final = GPTInferOut.model_validate(out_final_dict)
#         pred = int(out_final.pred_label)

#         t1 = now_ms()
#         return True, gold, pred, float(t1 - t0), None, pid

#     except Exception as e:
#         gold = int(row.get("label", 0)) if row.get("label") is not None else 0
#         return False, gold, None, 0.0, f"{type(e).__name__}: {e}", pid


# def worker_nothing_direct_one(dataset_name: str, row: dict):
#     """
#     debate/memory 없이 direct single-pass.
#     """
#     pid = os.getpid()
#     try:
#         dataset = DatasetEnum[dataset_name]
#         ok, target_text, gold, verr = _validate_row(row)
#         if not ok:
#             return False, gold, None, 0.0, f"ValueError: {verr}", pid

#         index, meta_train = get_index_meta(dataset)

#         similar_examples = search_similar_by_text(
#             config=G_CONFIG,
#             dataset=dataset,
#             index=index,
#             meta_train=meta_train,
#             query_text=target_text,
#             top_k=G_CONFIG.rag_top_k,
#         )

#         prompt = build_prompt_direct_nondebate(
#             target_text=target_text,
#             dataset=dataset,
#             similar_examples=similar_examples,
#         )

#         t0 = now_ms()
#         out_dict = G_CLIENTS["DIRECT"].call_api(prompt)
#         t1 = now_ms()

#         out = GPTInferOut.model_validate(out_dict)
#         pred = int(out.pred_label)

#         return True, gold, pred, float(t1 - t0), None, pid

#     except Exception as e:
#         gold = int(row.get("label", 0)) if row.get("label") is not None else 0
#         return False, gold, None, 0.0, f"{type(e).__name__}: {e}", pid


# # =========================================================
# # executor helpers
# # =========================================================
# def _build_executors(config: Config):
#     """
#     debate_memory.py와 동일 패턴:
#     - make_plan(config) -> [(gpu_id, n_workers), ...]
#     - gpu별 executor를 만들고 cycle로 라운드로빈 분배
#     """
#     plan = make_plan(config)
#     executors = []
#     for gid, n_workers in plan:
#         executors.append(
#             ProcessPoolExecutor(
#                 mp_context=MP_CTX,
#                 max_workers=n_workers,
#                 initializer=init_worker,
#                 initargs=(gid, config),
#             )
#         )
#     rr = itertools.cycle(executors)
#     return executors, rr


# def _shutdown_executors(executors):
#     for ex in executors:
#         ex.shutdown(wait=True, cancel_futures=False)


# # =========================================================
# # 1) DEBATE_DIRECT_TEST (parallel)
# # =========================================================
# def run_debate_direct_test(config: Config):
#     executors, rr = _build_executors(config)
#     try:
#         for dataset in TARGET_DATASETS:
#             tqdm_write(f"\n[debate_direct_test] dataset = {dataset.name}")

#             data_dir = Path(config.datasets_dir) / dataset.name
#             test_path = data_dir / "test.jsonl"
#             if not test_path.is_file():
#                 tqdm_write(f"  - test.jsonl not found, skip: {test_path}")
#                 continue

#             rows = list(load_jsonl(test_path))
#             if not rows:
#                 tqdm_write(f"  - no test samples: {test_path}")
#                 continue

#             futures = []
#             for row in rows:
#                 ex = next(rr)
#                 futures.append(ex.submit(worker_debate_direct_one, dataset.name, row))

#             ok_count = 0
#             fail_count = 0

#             y_true: list[int] = []
#             y_pred: list[int] = []
#             times_ms: list[float] = []

#             pbar = tqdm(
#                 desc=f"debate-direct {dataset.name}[test][parallel]",
#                 total=len(futures),
#                 unit="sample",
#                 dynamic_ncols=True,
#             )
#             pbar.set_postfix(ok=ok_count, fail=fail_count)

#             for fut in as_completed(futures):
#                 try:
#                     ok, gold, pred, time_ms, err, pid = fut.result()
#                 except Exception as e:
#                     fail_count += 1
#                     pbar.update(1)
#                     pbar.set_postfix(ok=ok_count, fail=fail_count)
#                     continue

#                 if not ok or pred is None:
#                     fail_count += 1
#                     # 필요하면 에러 로그 출력
#                     # tqdm_write(f"[debate_direct_test][FAIL] {dataset.name} | {err} | pid={pid}")
#                     pbar.update(1)
#                     pbar.set_postfix(ok=ok_count, fail=fail_count)
#                     continue

#                 y_true.append(int(gold))
#                 y_pred.append(int(pred))
#                 times_ms.append(float(time_ms))

#                 ok_count += 1
#                 pbar.update(1)
#                 pbar.set_postfix(ok=ok_count, fail=fail_count)

#             pbar.close()

#             if y_true:
#                 metrics = compute_binary_metrics(y_true, y_pred)
#                 save_metrics_csv_and_plots(
#                     config=config,
#                     dataset=dataset,
#                     mode="debate_direct_test",
#                     metrics=metrics,
#                 )

#                 time_info = {
#                     "n_samples": len(times_ms),
#                     "median_ms": float(np.median(times_ms)) if times_ms else 0.0,
#                     "mean_ms": float(np.mean(times_ms)) if times_ms else 0.0,
#                 }
#                 out_time = (
#                     Path(config.run_dir)
#                     / "debate_direct_test"
#                     / f"{dataset.name}_{config.api_model.name}_time.json"
#                 )
#                 out_time.parent.mkdir(parents=True, exist_ok=True)
#                 with out_time.open("w", encoding="utf-8") as f:
#                     json.dump(time_info, f, ensure_ascii=False, indent=2)

#                 tqdm_write(f"  - metrics & time saved for {dataset.name}")

#     finally:
#         _shutdown_executors(executors)


# # =========================================================
# # 2) NOTHING_DIRECT_TEST (parallel)
# # =========================================================
# def run_nothing_direct_test(config: Config):
#     executors, rr = _build_executors(config)
#     try:
#         for dataset in TARGET_DATASETS:
#             tqdm_write(f"\n[nothing_direct_test] dataset = {dataset.name}")

#             data_dir = Path(config.datasets_dir) / dataset.name
#             test_path = data_dir / "test.jsonl"
#             if not test_path.is_file():
#                 tqdm_write(f"  - test.jsonl not found, skip: {test_path}")
#                 continue

#             rows = list(load_jsonl(test_path))
#             if not rows:
#                 tqdm_write(f"  - no test samples: {test_path}")
#                 continue

#             futures = []
#             for row in rows:
#                 ex = next(rr)
#                 futures.append(ex.submit(worker_nothing_direct_one, dataset.name, row))

#             ok_count = 0
#             fail_count = 0

#             y_true: list[int] = []
#             y_pred: list[int] = []
#             times_ms: list[float] = []

#             pbar = tqdm(
#                 desc=f"nothing-direct {dataset.name}[test][parallel]",
#                 total=len(futures),
#                 unit="sample",
#                 dynamic_ncols=True,
#             )
#             pbar.set_postfix(ok=ok_count, fail=fail_count)

#             for fut in as_completed(futures):
#                 try:
#                     ok, gold, pred, time_ms, err, pid = fut.result()
#                 except Exception:
#                     fail_count += 1
#                     pbar.update(1)
#                     pbar.set_postfix(ok=ok_count, fail=fail_count)
#                     continue

#                 if not ok or pred is None:
#                     fail_count += 1
#                     # tqdm_write(f"[nothing_direct_test][FAIL] {dataset.name} | {err} | pid={pid}")
#                     pbar.update(1)
#                     pbar.set_postfix(ok=ok_count, fail=fail_count)
#                     continue

#                 y_true.append(int(gold))
#                 y_pred.append(int(pred))
#                 times_ms.append(float(time_ms))

#                 ok_count += 1
#                 pbar.update(1)
#                 pbar.set_postfix(ok=ok_count, fail=fail_count)

#             pbar.close()

#             if y_true:
#                 metrics = compute_binary_metrics(y_true, y_pred)
#                 save_metrics_csv_and_plots(
#                     config=config,
#                     dataset=dataset,
#                     mode="nothing_direct_test",
#                     metrics=metrics,
#                 )

#                 time_info = {
#                     "n_samples": len(times_ms),
#                     "median_ms": float(np.median(times_ms)) if times_ms else 0.0,
#                     "mean_ms": float(np.mean(times_ms)) if times_ms else 0.0,
#                 }
#                 out_time = (
#                     Path(config.run_dir)
#                     / "nothing_direct_test"
#                     / f"{dataset.name}_{config.api_model.name}_time.json"
#                 )
#                 out_time.parent.mkdir(parents=True, exist_ok=True)
#                 with out_time.open("w", encoding="utf-8") as f:
#                     json.dump(time_info, f, ensure_ascii=False, indent=2)

#                 tqdm_write(f"  - metrics & time saved for {dataset.name}")

#     finally:
#         _shutdown_executors(executors)
