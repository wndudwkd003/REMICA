# worker/rem_stage2.py

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from config.config import REM_STEP_1_DATASET, Config
from params.db_value import DB
from utils.client_utils import GPTClient, RemStage2Out
from utils.cuda_utils import split_workers
from utils.data_utils import JsonlDataset
from utils.db_utils import (
    exists,
    get_stage1,
    init_stage2_schema,
    open_db,
    upsert_stage2,
)
from utils.prompt_utils import build_rem_stage2_prompt

G_CLIENT = None

def init_worker(gpu_id, config: Config):
    global G_CLIENT

    G_CLIENT = GPTClient[RemStage2Out](
        model=config.gpt_model,
        max_output_tokens=config.gpt_max_output_tokens,
        schema=RemStage2Out,
        max_retries=config.max_retries,
        retry_sleep=0.5,
        temperature=config.gpt_temperature,
        top_p=config.gpt_top_p,
    )

def stage_2_job(job, db_path: str):
    global G_CLIENT

    ds, sid, text, label = job
    conn = open_db(db_path)

    try:
        rem_1_result = get_stage1(conn, sid)
        if rem_1_result is None:
            return (
                False,
                ds,
                sid,
                None,
                None,
                None,
                "NO_STAGE1_RESULT",
                os.getpid(),
            )

        stage1_pred = rem_1_result["pred_label"]
        stage1_rationale = rem_1_result["rationale"]

        prompt = build_rem_stage2_prompt(text, label, stage1_pred, stage1_rationale)
        out = G_CLIENT.call_api(prompt)

        evidence = out["evidence"]
        is_correct = int(label == stage1_pred)
        raw_json = json.dumps(out, ensure_ascii=False)

        return (
            True,
            ds,
            sid,
            is_correct,
            evidence,
            raw_json,
            None,
            os.getpid(),
        )

    except Exception as e:
        return (
            False,
            ds,
            sid,
            None,
            None,
            None,
            f"{type(e).__name__}: {str(e)}",
            os.getpid(),
        )
    finally:
        conn.close()


def run_rem_stage2(config: Config):
    split = config.rem_split
    print(f"[rem_stage2.py] split: {split}")

    db_path = config.remica_db_path

    conn = open_db(db_path)
    init_stage2_schema(conn)

    total_workers = config.rem_worker
    gpu_ids = config.rem_gpus
    plan = split_workers(total_workers, gpu_ids)
    print(f"[rem_stage2.py] gpu plan: {plan}  (sum={sum(n for _, n in plan)})")

    jobs = []

    for ds in REM_STEP_1_DATASET:
        ds_path = Path(config.datasets_dir) / ds.name / f"{split}.jsonl"
        print(f"[rem_stage2.py] Scanning dataset: {ds.name}, path: {ds_path}")

        dataset = JsonlDataset(ds_path, meta_to_text=False)

        for i in range(len(dataset)):
            sid, text, label, metadata = dataset[i]

            # Stage1에 없는 샘플은 스킵
            if not exists(conn, DB.REM_STAGE_1.value, sid):
                continue

            # Stage2에 이미 있는 샘플은 스킵
            if exists(conn, DB.REM_STAGE_2.value, sid):
                continue

            jobs.append((ds, sid, text, label))

    total = len(jobs)
    print(f"[rem_stage2.py] TODO jobs (Stage1 exists & Stage2 not exists): {total}")
    if total == 0:
        conn.close()
        print("[rem_stage2.py] All done. Exiting.")
        return db_path

    ok_count = 0
    fail_count = 0

    executors = []
    for gid, n_workers in plan:
        ex = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker,
            initargs=(gid, config),
        )
        executors.append(ex)

    futures = []
    for idx, job in enumerate(jobs):
        ex: ProcessPoolExecutor = executors[idx % len(executors)]
        futures.append(ex.submit(stage_2_job, job, db_path))

    pbar = tqdm(
        total=total, desc="[rem_stage2.py] stage2", unit="item", dynamic_ncols=True
    )

    try:
        for fut in as_completed(futures):
            try:
                (
                    ok,
                    ds,
                    sid,
                    is_correct,
                    evidence,
                    raw_json,
                    err,
                    pid,
                ) = fut.result()
            except Exception:
                fail_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)
                continue

            if not ok:
                fail_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)
                print(
                    f"[rem_stage2.py] FAIL {ds.value} | SID: {sid} | {err} | pid={pid}"
                )
                continue

            # Stage2 DB 저장
            upsert_stage2(
                conn,
                sid=sid,
                is_correct=is_correct,
                evidence=evidence,
                raw_json=raw_json,
            )

            print(
                f"[rem_stage2.py] OK   {ds.value} | SID: {sid} | Is_Correct: {is_correct} | {evidence} | pid={pid}"
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
        f"[rem_stage2.py] Done. ok={ok_count}, fail={fail_count}, total={total} | DB saved at: {db_path}"
    )
    return db_path
