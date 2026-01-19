# worker/rem_stag1.py

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from config.config import REM_STEP_1_DATASET, Config, DatasetEnum
from params.db_value import DB
from utils.data_utils import JsonlDataset

from utils.db_utils import open_db, init_stage1_schema, exists, upsert_stage1


from utils.gpt_client_stage1 import GPTClient
from utils.prompt_utils import build_rem_stage1_prompt
from utils.retriever import SimilarTextRetriever


# ---- 워커 프로세스에 1번만 만들 전역 객체 ----
_G_CLIENT = None
_G_RETRIEVER = None


def _init_worker(gpu_id, config: Config):

    global _G_CLIENT, _G_RETRIEVER

    if gpu_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config.emb_device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
        config.emb_device = "cuda"

    _G_CLIENT = GPTClient(
        model=config.gpt_model,
        max_output_tokens=config.gpt_max_output_tokens,
        max_retries=config.max_retries,
    )
    _G_RETRIEVER = SimilarTextRetriever(config)


def _run_one_stage1_job(job, config: Config):
    global _G_CLIENT, _G_RETRIEVER

    ds, sid, text, label = job
    try:
        sim_texts = _G_RETRIEVER.get_similar_texts(
            ds=ds,
            label=label,
            query_sid=sid,
            query_text=text,
            top_k=config.sim_k,
        )
        prompt = build_rem_stage1_prompt(text, ds, sim_texts)
        out = _G_CLIENT.call_api(prompt)

        pred_label = out["pred_label"]
        confidence = out["confidence"]
        rationale = out["rationale"]
        raw_json = json.dumps(out, ensure_ascii=False)

        return (
            True,
            ds,
            sid,
            label,
            pred_label,
            confidence,
            rationale,
            raw_json,
            None,
            os.getpid(),
        )

    except Exception as e:
        return (
            False,
            ds,
            sid,
            label,
            None,
            None,
            None,
            None,
            f"{type(e).__name__}: {e}",
            os.getpid(),
        )


def _split_workers(total_workers, gpu_ids):
    """
    예: total_workers=10, gpu_ids=[0,1,2,3] -> [(0,3),(1,3),(2,2),(3,2)]
    """
    g = len(gpu_ids)
    base, rem = divmod(total_workers, g)
    out = []
    for i, gid in enumerate(gpu_ids):
        n = base + (1 if i < rem else 0)
        out.append((gid, n))
    return out


def run_rem_stage1(config: Config):
    split = config.rem_split
    print(f"[rem_stage1.py] split: {split}")

    db_path = config.remica_db_path

    conn = open_db(db_path)
    init_stage1_schema(conn)

    total_workers = int(config.rem_worker)

    # 사용 GPU 리스트 (예: config.rem_gpus = [0,1,2,3])
    gpu_ids = config.rem_gpus
    use_gpu = len(gpu_ids) > 0

    if not use_gpu:
        # CPU retriever: executor 1개로만 돌려도 됨(프로세스 수는 total_workers)
        plan = [(None, total_workers)]
        print(f"[rem_stage1.py] retriever=CPU, workers={total_workers}")
    else:
        plan = _split_workers(total_workers, gpu_ids)
        print(f"[rem_stage1.py] gpu plan: {plan}  (sum={sum(n for _, n in plan)})")

    # 1) 메인에서 DB에 없는 것만 선별
    jobs = []
    scheduled = set()

    for ds in REM_STEP_1_DATASET:
        ds: DatasetEnum
        ds_path = Path(config.datasets_dir) / ds.name / f"{split}.jsonl"
        print(f"[rem_stage1.py] Scanning dataset: {ds.name}, path: {ds_path}")

        dataset = JsonlDataset(ds_path, meta_to_text=False)
        for i in range(len(dataset)):
            sid, text, label, metadata = dataset[i]

            if sid in scheduled:
                continue
            if exists(conn, DB.REM_STAGE_1.value, sid):
                continue

            scheduled.add(sid)
            jobs.append((ds, sid, text, label))

    total = len(jobs)
    print(f"[rem_stage1.py] TODO jobs (not in DB): {total}")
    if total == 0:
        conn.close()
        print(f"[rem_stage1.py] Nothing to do. DB at: {db_path}")
        return db_path

    ok_count = 0
    fail_count = 0

    # 2) GPU별 executor 여러 개 생성
    executors = []
    for gid, n_workers in plan:
        if n_workers <= 0:
            continue
        ex = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(gid, config),
        )
        executors.append(ex)

    if len(executors) == 0:
        conn.close()
        raise RuntimeError("No executors created. Check rem_worker/rem_gpus settings.")

    # 3) job을 executor들에 라운드로빈 분배
    futures = []
    for idx, job in enumerate(jobs):
        ex = executors[idx % len(executors)]
        futures.append(ex.submit(_run_one_stage1_job, job, config))

    pbar = tqdm(
        total=total, desc="[rem_stage1.py] stage1", unit="item", dynamic_ncols=True
    )
    try:
        for fut in as_completed(futures):
            try:
                (
                    ok,
                    ds,
                    sid,
                    true_label,
                    pred_label,
                    confidence,
                    rationale,
                    raw_json,
                    err,
                    pid,
                ) = fut.result()
            except Exception as e:
                fail_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)
                continue

            if not ok:
                fail_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)
                print(
                    f"[rem_stage1.py] FAIL {ds.value} | SID: {sid} | True: {true_label} | {err} | pid={pid}"
                )
                continue

            upsert_stage1(conn, sid, pred_label, confidence, rationale, raw_json)
            print(
                f"[rem_stage1.py] OK   {ds.value} | SID: {sid} | True: {true_label} | Pred: {pred_label} | Conf: {confidence:.4f} | {rationale} | pid={pid}"
            )
            conn.commit()

            ok_count += 1
            pbar.update(1)
            pbar.set_postfix(ok=ok_count, fail=fail_count)
    finally:
        pbar.close()
        for ex in executors:
            ex.shutdown(wait=False)
        conn.close()

    print(
        f"[rem_stage1.py] Done. ok={ok_count}, fail={fail_count}, total={total} | DB saved at: {db_path}"
    )
    return db_path
