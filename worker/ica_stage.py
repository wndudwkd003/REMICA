# worker/ica.py

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from config.config import ICA_STEP_DATASET, Config, ContextDatasetEnum
from utils.db_utils import exists, init_ica_schema, open_db, upsert_ica
from utils.gpt_client_ica import GPTClientICA
from utils.prompt_utils import build_ica_prompt

_G_CLIENT = None


def _init_worker(gpu_id, config: Config):
    global _G_CLIENT

    if gpu_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config.emb_device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
        config.emb_device = "cuda"

    _G_CLIENT = GPTClientICA(
        model=config.gpt_model,
        max_output_tokens=config.gpt_max_output_tokens,
        max_retries=config.max_retries,
    )


def _split_workers(total_workers: int, gpu_ids: list[int]):
    g = len(gpu_ids)
    base, rem = divmod(total_workers, g)
    out = []
    for i, gid in enumerate(gpu_ids):
        n = base + (1 if i < rem else 0)
        out.append((gid, n))
    return out


def _make_4turn_windows(turns):
    n = len(turns)
    out = []
    if n < 4:
        return out
    for s in range(0, n - 4 + 1):
        out.append(turns[s : s + 4])
    return out


def _run_one_ica_job(job, config: Config):
    global _G_CLIENT

    (
        ds,
        split,
        sid,
        conversation_label,
        turns4,
        source_cid,
        source_file,
        window_start,
        window_end,
        last_text,
    ) = job

    try:
        prompt = build_ica_prompt(turns4, source_dataset=ds.value, split=split, sid=sid)
        out = _G_CLIENT.call_api(prompt)

        context_summary = out["context_summary"]
        triggers_json = json.dumps(out["triggers"], ensure_ascii=False)
        targets_json = json.dumps(out["targets"], ensure_ascii=False)
        rules_json = json.dumps(out["rules"], ensure_ascii=False)
        raw_json = json.dumps(out, ensure_ascii=False)

        return (
            True,
            sid,
            ds.value,
            split,
            int(conversation_label),
            context_summary,
            triggers_json,
            targets_json,
            rules_json,
            raw_json,
            source_cid,
            source_file,
            int(window_start),
            int(window_end),
            last_text,
            os.getpid(),
            None,
        )

    except Exception as e:
        return (
            False,
            sid,
            ds.value,
            split,
            int(conversation_label),
            None,
            None,
            None,
            None,
            None,
            source_cid,
            source_file,
            int(window_start),
            int(window_end),
            last_text,
            os.getpid(),
            f"{type(e).__name__}: {e}",
        )


def run_ica(config: Config):
    split = config.rem_split
    print(f"[ica.py] split: {split}")

    db_path = config.remica_db_path
    conn = open_db(db_path)
    init_ica_schema(conn)

    total_workers = int(config.rem_worker)
    gpu_ids = list(config.rem_gpus)
    use_gpu = len(gpu_ids) > 0

    if not use_gpu:
        plan = [(None, total_workers)]
        print(f"[ica.py] retriever=CPU, workers={total_workers}")
    else:
        plan = _split_workers(total_workers, gpu_ids)
        print(f"[ica.py] gpu plan: {plan} (sum={sum(n for _, n in plan)})")

    jobs = []
    scheduled = set()

    for ds in ICA_STEP_DATASET:
        ds: ContextDatasetEnum
        ds_path = Path(config.context_datasets_dir) / ds.value / f"{split}.jsonl"
        print(f"[ica.py] Scanning dataset: {ds.value}, path: {ds_path}")

        with ds_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)

                cid = obj["cid"]
                turns = obj["turns"]

                windows = _make_4turn_windows(turns)
                for w_idx, turns4 in enumerate(windows):
                    last_turn = turns4[-1]
                    conversation_label = int(last_turn["label"])

                    sid = f"{cid}#w{w_idx:04d}"
                    if sid in scheduled:
                        continue
                    if exists(conn, "ica", sid):
                        continue

                    # ✅ upsert_ica가 요구하는 메타 5개를 여기서 같이 만든다
                    source_cid = cid
                    source_file = str(ds_path)  # 원본 jsonl 경로
                    window_start = int(w_idx)  # windows 시작 index
                    window_end = int(w_idx + 4)  # exclusive (권장)
                    last_text = str(last_turn["text"]).strip()  # 마지막 턴 텍스트

                    scheduled.add(sid)
                    jobs.append(
                        (
                            ds,
                            split,
                            sid,
                            conversation_label,
                            turns4,
                            source_cid,
                            source_file,
                            window_start,
                            window_end,
                            last_text,
                        )
                    )

    total = len(jobs)
    print(f"[ica.py] TODO jobs (not in DB): {total}")
    if total == 0:
        conn.close()
        print(f"[ica.py] Nothing to do. DB at: {db_path}")
        return db_path

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

    futures = []
    for idx, job in enumerate(jobs):
        ex = executors[idx % len(executors)]
        futures.append(ex.submit(_run_one_ica_job, job, config))

    ok_count = 0
    fail_count = 0

    pbar = tqdm(total=total, desc="[ica.py] ica", unit="item", dynamic_ncols=True)
    try:
        for fut in as_completed(futures):
            (
                ok,
                sid,

                conversation_label,
                context_summary,
                triggers_json,
                targets_json,
                rules_json,
                raw_json,
                source_cid,
                source_file,
                window_start,
                window_end,
                last_text,
                pid,
                err,
            ) = fut.result()

            if not ok:
                fail_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)
                print(
                    f"[ica.py] FAIL {source_dataset} | SID: {sid} | label={conversation_label} | err={err} | pid={pid}"
                )
                continue

            upsert_ica(
                conn,
                sid=sid,
                source_dataset=source_dataset,
                split=split,
                conversation_label=conversation_label,
                context_summary=context_summary,
                triggers_json=triggers_json,
                targets_json=targets_json,
                rules_json=rules_json,
                raw_json=raw_json,
                # ✅ upsert_ica 신규 필수 인자들
                source_cid=source_cid,
                source_file=source_file,
                window_start=window_start,
                window_end=window_end,
                last_text=last_text,
            )
            conn.commit()

            ok_count += 1
            pbar.update(1)
            pbar.set_postfix(ok=ok_count, fail=fail_count)

            print(
                f"[ica.py] OK {source_dataset} | SID: {sid} | label={conversation_label} | "
                f"summary={context_summary[:80]} | rules={rules_json} | pid={pid}"
            )

    finally:
        pbar.close()
        for ex in executors:
            ex.shutdown(wait=False)
        conn.close()

    print(
        f"[ica.py] Done. ok={ok_count}, fail={fail_count}, total={total} | DB: {db_path}"
    )
    return db_path
