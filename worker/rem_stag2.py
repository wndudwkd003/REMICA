# worker/rem_stag2.py
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from config.config import Config, REM_STEP_1_DATASET, DatasetEnum
from params.db_value import DB
from utils.data_utils import JsonlDataset
from utils.db_utils import open_db, init_stage1_schema, init_stage2_schema
from utils.retriever import SimilarTextRetriever
from utils.prompt_utils import build_rem_stage2_prompt
from utils.gpt_client_stage2 import GPTClientStage2


# ---- 워커 프로세스에 1번만 만들 전역 객체 ----
_G_CLIENT2 = None
_G_RETRIEVER = None


def _init_worker(gpu_id, config: Config):
    global _G_CLIENT2, _G_RETRIEVER

    if gpu_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config.emb_device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
        config.emb_device = "cuda"

    _G_CLIENT2 = GPTClientStage2(
        model=config.gpt_model,
        max_output_tokens=config.gpt_max_output_tokens,
        max_retries=config.max_retries,
    )
    _G_RETRIEVER = SimilarTextRetriever(config)


def _run_one_stage2_job(job, config: Config):
    """
    job: (ds, sid, text, true_label, stage1_pred, stage1_rationale)
    - DB 접근 금지
    """
    global _G_CLIENT2, _G_RETRIEVER

    ds, sid, text, true_label, stage1_pred, stage1_rationale = job

    try:
        true_label = int(true_label)
        stage1_pred = int(stage1_pred)

        is_correct = stage1_pred == true_label

        # neighbors 풀 선택:
        # - 정답: true_label 풀
        # - 오답: stage1_pred 풀 (오답이 왜 나왔는지 근처에서 패턴을 보게)
        label_for_neighbors = true_label if is_correct else stage1_pred

        sim_texts = _G_RETRIEVER.get_similar_texts(
            ds=ds,
            label=label_for_neighbors,
            query_sid=sid,
            query_text=text,
            top_k=config.sim_k,
        )

        prompt = build_rem_stage2_prompt(
            ds=ds,
            text=text,
            similar_texts=sim_texts,
            true_label=true_label,
            stage1_pred=stage1_pred,
            stage1_rationale=str(stage1_rationale),
        )

        out = _G_CLIENT2.call_api(prompt)  # {"evidence":..., "memory":...}

        return (
            True,
            ds,
            sid,
            true_label,
            stage1_pred,
            int(is_correct),
            out["evidence"],
            out["memory"],
            json.dumps(out, ensure_ascii=False),
            None,
            os.getpid(),
        )

    except Exception as e:
        return (
            False,
            ds,
            sid,
            int(true_label),
            int(stage1_pred),
            None,
            None,
            None,
            None,
            f"{type(e).__name__}: {e}",
            os.getpid(),
        )


def _split_workers(total_workers, gpu_ids):
    g = len(gpu_ids)
    base, rem = divmod(total_workers, g)
    out = []
    for i, gid in enumerate(gpu_ids):
        n = base + (1 if i < rem else 0)
        out.append((gid, n))
    return out


def run_rem_stage2(config: Config):
    split = config.rem_split
    print(f"[rem_stage2.py] split: {split}")

    rem_dir = Path(config.rem_dir)
    db1_path = rem_dir / "stage_1.sqlite3"
    db2_path = rem_dir / "stage_2.sqlite3"

    conn1 = open_db(db1_path)
    init_stage1_schema(conn1)

    conn2 = open_db(db2_path)
    init_stage2_schema(conn2)

    total_workers = int(config.rem_worker)

    gpu_ids = getattr(config, "rem_gpus", [])
    use_gpu = len(gpu_ids) > 0

    if not use_gpu:
        plan = [(None, total_workers)]
        print(f"[rem_stage2.py] retriever=CPU, workers={total_workers}")
    else:
        plan = _split_workers(total_workers, gpu_ids)
        print(f"[rem_stage2.py] gpu plan: {plan}  (sum={sum(n for _, n in plan)})")

    # ---- 1) 메인에서 jobs 선별 ----
    jobs = []
    scheduled = set()

    for ds in REM_STEP_1_DATASET:
        ds: DatasetEnum
        ds_path = Path(config.datasets_dir) / ds.name / f"{split}.jsonl"
        print(f"[rem_stage2.py] Scanning dataset: {ds.name}, path: {ds_path}")

        dataset = JsonlDataset(ds_path, meta_to_text=False)

        for i in range(len(dataset)):
            sid, text, true_label, metadata = dataset[i]
            true_label = int(true_label)

            if sid in scheduled:
                continue

            # stage2 이미 있으면 skip
            if exists(conn2, DB.REM_STAGE_2.value, sid):
                continue

            # stage1 결과 없으면 skip
            s1 = get_stage1(conn1, sid)
            if s1 is None:
                continue

            stage1_pred = int(s1["pred_label"])
            stage1_rationale = s1["rationale"]

            scheduled.add(sid)
            jobs.append((ds, sid, text, true_label, stage1_pred, stage1_rationale))

    total = len(jobs)
    print(f"[rem_stage2.py] TODO jobs (stage1 exists, not in stage2): {total}")

    if total == 0:
        conn1.close()
        conn2.close()
        print(f"[rem_stage2.py] Nothing to do. DB at: {db2_path}")
        return db2_path

    ok_count = 0
    fail_count = 0

    # ---- 2) executor 생성 ----
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
        conn1.close()
        conn2.close()
        raise RuntimeError("No executors created. Check rem_worker/rem_gpus settings.")

    # ---- 3) jobs 라운드로빈 분배 ----
    futures = []
    for idx, job in enumerate(jobs):
        ex = executors[idx % len(executors)]
        futures.append(ex.submit(_run_one_stage2_job, job, config))

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
                    true_label,
                    stage1_pred,
                    is_correct,
                    evidence,
                    memory,
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

            # DB 저장 (새 스키마/시그니처 기준)
            upsert_stage2(
                conn2,
                sid=sid,
                true_label=int(true_label),
                stage1_pred=int(stage1_pred),
                is_correct=int(is_correct),
                evidence=str(evidence),
                memory=str(memory),
                raw_json=raw_json,
            )
            conn2.commit()

            print(
                f"[rem_stage2.py] OK {ds.value} | SID: {sid} | true={true_label} | pred={stage1_pred} | correct={is_correct} | {evidence} | pid={pid}"
            )

            ok_count += 1
            pbar.update(1)
            pbar.set_postfix(ok=ok_count, fail=fail_count)

    finally:
        pbar.close()
        for ex in executors:
            ex.shutdown(wait=False)
        conn1.close()
        conn2.close()

    print(
        f"[rem_stage2.py] Done. ok={ok_count}, fail={fail_count}, total={total} | DB saved at: {db2_path}"
    )
    return db2_path
