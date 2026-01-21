# worker/rem_stag2.py
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore
from tqdm.auto import tqdm

from config.config import Config, DatasetEnum
from params.db_value import DB
from utils.data_utils import JsonlDataset
from utils.db_utils import (
    exists,
    get_ica,
    get_stage1,
    init_stage2_schema,
    open_db,
    upsert_stage2,
)
from utils.gpt_client_stage2 import GPTClientStage2
from utils.prompt_utils import build_rem_stage2_prompt


G_CLIENT = None
G_DB = None
G_ICA_RETRIEVER = None


class ICARuleRetriever:
    def __init__(
        self,
        db_path: str,
        split: str,
        emb_model: str,
        device: str,
        faiss_dir: str = "rem/faiss",
    ):
        self.db_path = str(db_path)
        self.split = str(split)
        self.device = str(device)

        faiss_dir_p = Path(faiss_dir)
        self.index_path = faiss_dir_p / f"ica_last_text_{split}.index"
        self.meta_path = faiss_dir_p / f"ica_last_text_{split}.meta.jsonl"

        if not self.index_path.exists():
            raise FileNotFoundError(f"missing_faiss_index: {self.index_path}")
        if not self.meta_path.exists():
            raise FileNotFoundError(f"missing_faiss_meta: {self.meta_path}")

        self.model = SentenceTransformer(str(emb_model), device=device)
        self.index = faiss.read_index(str(self.index_path))

        self.meta = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("sid"):
                    self.meta.append(obj)

    def embed(self, text: str) -> np.ndarray:
        t = (text or "").strip()
        if not t:
            t = " "
        v = self.model.encode([t], normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def search(
        self,
        query_text: str,
        true_label: int | None,
        top_k: int,
        conn,
        label_filter_strict: bool = True,
        oversample: int = 5,
    ) -> list[dict]:
        if top_k <= 0:
            return []
        if self.index.ntotal <= 0:
            return []
        if len(self.meta) <= 0:
            return []

        k = max(int(top_k) * int(oversample), int(top_k))
        q = self.embed(query_text)

        try:
            _, I = self.index.search(q, k)
        except Exception:
            return []

        hits = []
        for idx in I[0].tolist():
            if idx < 0:
                continue
            if idx >= len(self.meta):
                continue
            m = self.meta[idx]
            sid = m.get("sid")
            if not sid:
                continue

            if true_label is not None and label_filter_strict:
                try:
                    if int(m.get("conversation_label", -1)) != int(true_label):
                        continue
                except Exception:
                    continue

            row = get_ica(conn, sid)
            if row is None:
                continue
            row["ica_sid"] = sid
            hits.append(row)
            if len(hits) >= int(top_k):
                break

        if len(hits) == 0 and true_label is not None and label_filter_strict:
            return self.search(
                query_text=query_text,
                true_label=true_label,
                top_k=top_k,
                conn=conn,
                label_filter_strict=False,
                oversample=oversample,
            )

        return hits


def label_stability(run_preds: list[int]) -> float:
    if not run_preds:
        return 0.0
    arr = np.array([int(x) for x in run_preds], dtype=np.int64)
    _, counts = np.unique(arr, return_counts=True)
    if len(counts) == 0:
        return 0.0
    return float(np.max(counts) / len(arr))


def evidence_consistency(model: SentenceTransformer, evidences: list[str]) -> float:
    evs = [str(e).strip() for e in (evidences or []) if str(e).strip()]
    if len(evs) <= 1:
        return 0.0
    try:
        embs = model.encode(evs, normalize_embeddings=True)
    except Exception:
        return 0.0
    embs = np.asarray(embs, dtype=np.float32)
    if embs.ndim != 2 or embs.shape[0] <= 1:
        return 0.0
    sims = []
    for i in range(embs.shape[0]):
        for j in range(i + 1, embs.shape[0]):
            sims.append(float(np.dot(embs[i], embs[j])))
    if not sims:
        return 0.0
    m = float(np.mean(sims))
    m = max(-1.0, min(1.0, m))
    return float(0.5 * (m + 1.0))


def calc_reliability(
    model: SentenceTransformer,
    run_preds: list[int],
    run_evidences: list[str],
    alpha: float = 0.5,
) -> float:
    a = float(alpha)
    a = max(0.0, min(1.0, a))
    ls = label_stability(run_preds)
    ec = evidence_consistency(model, run_evidences)
    r = a * ls + (1.0 - a) * ec
    return float(max(0.0, min(1.0, r)))


def build_rules_run2(rules: list[str]) -> list[str]:
    rr = [str(r).strip() for r in (rules or []) if str(r).strip()]
    if not rr:
        return []
    rr = rr[1:]
    return rr[::2]


def init_worker(gpu_id, config: Config):
    global G_CLIENT, G_DB, G_ICA_RETRIEVER

    if gpu_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
        device = "cuda"

    G_DB = open_db(config.remica_db_path)

    G_CLIENT = GPTClientStage2(
        model=config.gpt_model,
        max_output_tokens=config.gpt_max_output_tokens,
        max_retries=config.max_retries,
    )

    G_ICA_RETRIEVER = ICARuleRetriever(
        db_path=config.remica_db_path,
        split=config.rem_split,
        emb_model=config.emb_model,
        device=device,
        faiss_dir="rem/faiss",
    )


def run_one_stage2_job(job, config: Config):
    global G_CLIENT, G_DB, G_ICA_RETRIEVER

    ds, sid, text, true_label = job

    if G_DB is None or G_ICA_RETRIEVER is None or G_CLIENT is None:
        return (
            False,
            ds.value,
            sid,
            int(true_label),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            os.getpid(),
            "not_initialized",
        )

    try:
        s1 = get_stage1(G_DB, sid)
        if s1 is None:
            raise RuntimeError("missing_stage1")

        stage1_pred_label = int(s1.get("pred_label", 0))
        stage1_rationale = str(s1.get("rationale", "") or "").strip()

        ica_rows = G_ICA_RETRIEVER.search(
            query_text=str(text),
            true_label=int(true_label),
            top_k=int(getattr(config, "sim_k", 3)),
            conn=G_DB,
        )

        rules_pool = []
        for r in ica_rows:
            s = r.get("rules_json") or "[]"
            try:
                rules = json.loads(s)
            except Exception:
                rules = []
            if isinstance(rules, list):
                for x in rules:
                    if isinstance(x, str) and x.strip():
                        rules_pool.append(x.strip())

        seen = set()
        rules_pool_uniq = []
        for x in rules_pool:
            if x in seen:
                continue
            seen.add(x)
            rules_pool_uniq.append(x)

        runs = []

        runs.append(
            {
                "run_tag": "run0",
                "pred_label": int(stage1_pred_label),
                "evidence": stage1_rationale,
                "memory": "",
                "used_rules": [],
            }
        )

        for run_tag in ["run1", "run2"]:
            if run_tag == "run1":
                ica_rules = rules_pool_uniq
            else:
                ica_rules = build_rules_run2(rules_pool_uniq)

            prompt = build_rem_stage2_prompt(
                text=str(text),
                run_tag=str(run_tag),
                ica_rules=ica_rules,
                stage1_pred_label=int(stage1_pred_label),
                stage1_rationale=str(stage1_rationale),
                true_label=int(true_label),
            )

            out = G_CLIENT.call_api(prompt)

            runs.append(
                {
                    "run_tag": str(out.get("run_tag", run_tag)),
                    "pred_label": int(out["pred_label"]),
                    "evidence": str(out.get("evidence", "") or ""),
                    "memory": str(out.get("memory", "") or ""),
                    "used_rules": list(out.get("used_rules", []) or []),
                }
            )

        run_preds = [int(r.get("pred_label", 0)) for r in runs]
        run_evidences = [str(r.get("evidence", "") or "") for r in runs]
        reliability = calc_reliability(
            G_ICA_RETRIEVER.model, run_preds, run_evidences, alpha=0.5
        )

        pick = next((r for r in runs if r.get("run_tag") == "run1"), runs[0])
        final_pred = int(pick.get("pred_label", stage1_pred_label))
        is_correct = 1 if final_pred == int(true_label) else 0

        runs_json = json.dumps(runs, ensure_ascii=False)
        raw_json = json.dumps(
            {
                "sid": sid,
                "dataset": ds.value,
                "true_label": int(true_label),
                "stage1": s1,
                "ica_topk_sids": [r.get("ica_sid") for r in ica_rows],
                "rules_pool": rules_pool_uniq,
                "runs": runs,
                "reliability": float(reliability),
                "final_pick": str(pick.get("run_tag", "run1")),
            },
            ensure_ascii=False,
        )

        return (
            True,
            ds.value,
            sid,
            int(true_label),
            int(final_pred),
            int(is_correct),
            str(pick.get("evidence", "") or ""),
            str(pick.get("memory", "") or ""),
            float(reliability),
            runs_json,
            raw_json,
            os.getpid(),
            None,
        )

    except Exception as e:
        return (
            False,
            ds.value,
            sid,
            int(true_label),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            os.getpid(),
            f"{type(e).__name__}: {e}",
        )


def split_workers(total_workers: int, gpu_ids: list[int]):
    g = len(gpu_ids)
    if g <= 0:
        return [(None, int(total_workers))]
    base, rem = divmod(int(total_workers), g)
    out = []
    for i, gid in enumerate(gpu_ids):
        n = base + (1 if i < rem else 0)
        out.append((gid, n))
    return out


def run_rem_stage2(config: Config):
    split = config.rem_split
    print(f"[rem_stag2.py] split: {split}")

    db_path = config.remica_db_path
    conn = open_db(db_path)
    init_stage2_schema(conn)
    conn.commit()

    total_workers = int(config.rem_worker)
    gpu_ids = list(getattr(config, "rem_gpus", []) or [])
    use_gpu = len(gpu_ids) > 0

    plan = split_workers(total_workers, gpu_ids) if use_gpu else [(None, total_workers)]

    jobs = []
    scheduled = set()

    for ds in config.rem_step1_datasets:
        ds: DatasetEnum
        ds_path = Path(config.datasets_dir) / ds.name / f"{split}.jsonl"
        if not ds_path.exists():
            print(f"[rem_stag2.py] SKIP missing dataset file: {ds_path}")
            continue

        dataset = JsonlDataset(ds_path, meta_to_text=config.meta_to_text)

        for i in range(len(dataset)):
            sid, text, true_label, metadata = dataset[i]

            if sid in scheduled:
                continue
            if exists(conn, DB.REM_STAGE_2.value, sid):
                continue

            s1 = get_stage1(conn, sid)
            if s1 is None:
                continue

            scheduled.add(sid)
            jobs.append((ds, sid, text, int(true_label)))

    total = len(jobs)
    print(f"[rem_stag2.py] TODO jobs (not in DB): {total}")
    if total == 0:
        conn.close()
        return db_path

    executors = []
    for gid, n_workers in plan:
        if int(n_workers) <= 0:
            continue
        executors.append(
            ProcessPoolExecutor(
                max_workers=int(n_workers),
                initializer=init_worker,
                initargs=(gid, config),
            )
        )

    if len(executors) == 0:
        conn.close()
        raise RuntimeError("no_executors")

    futures = []
    for idx, job in enumerate(jobs):
        ex = executors[idx % len(executors)]
        futures.append(ex.submit(run_one_stage2_job, job, config))

    ok_count = 0
    fail_count = 0

    pbar = tqdm(
        total=total, desc="[rem_stag2.py] stage2", unit="item", dynamic_ncols=True
    )
    try:
        for fut in as_completed(futures):
            try:
                (
                    ok,
                    ds_name,
                    sid,
                    true_label,
                    pred_label,
                    is_correct,
                    evidence,
                    memory,
                    reliability,
                    runs_json,
                    raw_json,
                    pid,
                    err,
                ) = fut.result()
            except Exception as e:
                fail_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)
                print(f"[rem_stag2.py] FAIL unknown | err={type(e).__name__}: {e}")
                continue

            if not ok:
                fail_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)
                print(
                    f"[rem_stag2.py] FAIL {ds_name} | SID: {sid} | true={true_label} | err={err} | pid={pid}"
                )
                continue

            upsert_stage2(
                conn,
                sid=sid,
                true_label=int(true_label),
                pred_label=int(pred_label),
                is_correct=int(is_correct),
                evidence=str(evidence),
                memory=str(memory),
                reliability=float(reliability),
                runs_json=str(runs_json),
                raw_json=str(raw_json),
            )
            conn.commit()

            ok_count += 1
            pbar.update(1)
            pbar.set_postfix(ok=ok_count, fail=fail_count)

            ev = (str(evidence) or "").replace("\n", " ").strip()
            if len(ev) > 160:
                ev = ev[:160] + "..."
            print(
                f"[rem_stag2.py] OK {ds_name} | SID: {sid} | true={true_label} pred={pred_label} "
                f"correct={is_correct} rel={float(reliability):.3f} | evidence={ev} | pid={pid}"
            )

    finally:
        pbar.close()
        for ex in executors:
            ex.shutdown(wait=False)
        conn.close()

    print(
        f"[rem_stag2.py] Done. ok={ok_count}, fail={fail_count}, total={total} | DB: {db_path}"
    )
    return db_path
