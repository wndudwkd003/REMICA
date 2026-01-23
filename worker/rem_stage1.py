# worker/rem_stage1.py

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from config.config import REM_STEP_1_DATASET, Config
from params.db_value import DB
from utils.client_utils import GPTClient, RemStage1Out
from utils.cuda_utils import split_workers
from utils.data_utils import JsonlDataset
from utils.db_utils import exists, init_stage1_schema, open_db, upsert_stage1
from utils.prompt_utils import build_dataset_perspective, build_rem_stage1_prompt
from utils.retriever import SimilarTextRetriever

# ---- 워커 프로세스에 1번만 만들 전역 객체 ----
G_CLIENT = None
G_RETRIEVER = None


def init_worker(gpu_id, config: Config):
    global G_CLIENT, G_RETRIEVER

    G_CLIENT = GPTClient[RemStage1Out](
        model=config.gpt_model,
        max_output_tokens=config.gpt_max_output_tokens,
        max_retries=config.max_retries,
        top_p=config.gpt_top_p,
        temperature=config.gpt_temperature,
        schema=RemStage1Out,
    )
    G_RETRIEVER = SimilarTextRetriever(config, device=f"cuda:{gpu_id}")


def stage_1_job(job, config: Config):
    global G_CLIENT, G_RETRIEVER

    ds, sid, text, label = job
    try:
        ds_perspective = build_dataset_perspective(ds)

        sim_examples = G_RETRIEVER.get_similar_texts(
            ds=ds,                 # DatasetEnum 그대로 넘겨도 됩니다
            label=label,
            query_sid=sid,
            query_text=text,
            top_k=config.rem_1_top_k,
        )


        prompt = build_rem_stage1_prompt(text, ds_perspective, sim_examples)
        out = G_CLIENT.call_api(prompt)

        pred_label = out["pred_label"]
        rationale = out["rationale"]
        raw_json = json.dumps(out, ensure_ascii=False)

        return (
            True,
            ds,
            sid,
            label,
            pred_label,
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
            f"{type(e).__name__}: {e}",
            os.getpid(),
        )

def run_rem_stage1(config: Config):
    split = config.rem_split
    print(f"[rem_stage1.py] split: {split}")

    db_path = config.remica_db_path

    conn = open_db(db_path)
    init_stage1_schema(conn)

    total_workers = config.rem_worker

    # 사용 GPU 리스트 (예: config.rem_gpus = [0,1,2,3])
    gpu_ids = config.rem_gpus
    plan = split_workers(total_workers, gpu_ids)
    print(f"[rem_stage1.py] gpu plan: {plan}  (sum={sum(n for _, n in plan)})")

    # 1) 메인에서 DB에 없는 것만 선별
    jobs = []

    for ds in REM_STEP_1_DATASET:
        ds_path = Path(config.datasets_dir) / ds.name / f"{split}.jsonl"
        print(f"[rem_stage1.py] Scanning dataset: {ds.name}, path: {ds_path}")

        dataset = JsonlDataset(ds_path, meta_to_text=False)
        for i in range(len(dataset)):
            sid, text, label, metadata = dataset[i]

            # DB에 존재하는것은 스킵
            if exists(conn, DB.REM_STAGE_1.value, sid):
                continue

            jobs.append((ds, sid, text, label))

    total = len(jobs)
    # 이미 DB에 다 있으면 종료
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
        ex = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker,
            initargs=(gid, config),
        )
        executors.append(ex)

    # 3) job을 executor들에 라운드로빈 분배
    futures = []
    for idx, job in enumerate(jobs):
        ex: ProcessPoolExecutor = executors[idx % len(executors)]
        futures.append(ex.submit(stage_1_job, job, config))

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
                    rationale,
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
                    f"[rem_stage1.py] FAIL {ds.value} | SID: {sid} | True: {true_label} | {err} | pid={pid}"
                )
                continue

            upsert_stage1(conn, sid, pred_label,  rationale, raw_json)
            print(
                f"[rem_stage1.py] OK   {ds.value} | SID: {sid} | True: {true_label} | Pred: {pred_label} | {rationale} | pid={pid}"
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
