# worker/save_memory.py


import itertools
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

from config.config import ChainEnum, Config, DatasetEnum
from utils.faiss_utils import (
    search_similar_by_text,
    pick_topk_with_both_labels_in_order,
)
from utils.gpu_utils import make_plan
from utils.prompt_builder_debate import build_prompt_chain_step
import utils.process_utils as pu
from utils.path_utils import get_memory_root

MP_CTX = mp.get_context("spawn")


def tqdm_write(msg: str) -> None:
    tqdm.write(msg)


def pretty_json(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)


def log_agent(sample_id: str, agent_name: str, prompt: str, out_dict: dict) -> None:
    sep = "=" * 120
    tqdm_write("\n" + sep)
    tqdm_write(f"[DEBATE][{sample_id}][{agent_name}] PROMPT")
    tqdm_write("-" * 120)
    tqdm_write(prompt)
    tqdm_write("-" * 120)
    tqdm_write(f"[DEBATE][{sample_id}][{agent_name}] OUTPUT(JSON)")
    tqdm_write("-" * 120)
    tqdm_write(pretty_json(out_dict))
    tqdm_write(sep + "\n")


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


def load_done_ids(jsonl_path: Path) -> set[str]:
    done = set()
    if not jsonl_path.is_file():
        return done

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(str(rec["id"]))
    return done


def worker_save_one(dataset_name: str, row: dict):
    pid = os.getpid()
    sample_id = str(row["id"])

    try:
        dataset = DatasetEnum[dataset_name]
        index, meta_train = pu.get_index_meta(dataset)

        text = row["text"]
        label = int(row["label"])

        use_k = pu.G_CONFIG.rag_top_k
        cand_k = use_k * pu.G_CONFIG.rag_cand_k

        similar_examples = search_similar_by_text(
            config=pu.G_CONFIG,
            dataset=dataset,
            index=index,
            meta_train=meta_train,
            query_text=text,
            top_k=cand_k,
            query_id=sample_id,
        )

        if pu.G_CONFIG.is_sim_legacy == False:
            similar_examples = pick_topk_with_both_labels_in_order(
                candidates_full=similar_examples,
                use_k=use_k,
            )

        out_A1 = None
        out_B1 = None
        out_A2 = None
        out_B2 = None

        mem_rec = {
            "id": sample_id,
            "text": text,
            "label": label,
            "chain_mode": pu.G_CONFIG.chain_mode.value,
            "actual_label_intervention": pu.G_CONFIG.actual_label_intervention,
        }

        for step_index in (1, 2, 3, 4):
            step_name, prompt = build_prompt_chain_step(
                step_index=step_index,
                target_text=text,
                dataset=dataset,
                similar_examples=similar_examples,
                out_A1=out_A1,
                out_B1=out_B1,
                out_A2=out_A2,
                out_B2=out_B2,
                actual_label_intervention=pu.G_CONFIG.actual_label_intervention,
                actual_label=label,
                chain_mode=pu.G_CONFIG.chain_mode,
            )

            client_key = "A" if step_name.startswith("A") else "B"
            out_raw = pu.G_CLIENTS[client_key].call_api(prompt)
            out_dict = pu.validate_step_output(step_name, out_raw)

            mem_rec[step_name] = {"prompt": prompt, "output": out_dict}

            if step_name == "A1":
                out_A1 = out_dict
            elif step_name == "B1":
                out_B1 = out_dict
            elif step_name == "A2":
                out_A2 = out_dict
            elif step_name == "B2":
                out_B2 = out_dict

        return True, mem_rec, None, pid

    except Exception as e:
        return False, {"id": sample_id}, f"{type(e).__name__}: {e}", pid


def run_save_memory(config: Config) -> None:
    if config.chain_mode == ChainEnum.NONE:
        raise ValueError(
            "chain_mode == NONE 인 경우는 메모리 저장이 아니라 faiss_index 기반 RAG만 사용합니다."
        )

    root = get_memory_root(config)
    tqdm_write(f"[run_save_memory] memory_root = {root}")

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
            tqdm_write(f"\n[save_memory] dataset = {dataset.name}")

            train_path = Path(config.datasets_dir) / dataset.name / "train.jsonl"
            if not train_path.is_file():
                tqdm_write(f"  - train.jsonl not found, skip: {train_path}")
                continue

            out_path = root / f"{dataset.name}.jsonl"
            done_ids = load_done_ids(out_path)

            futures = []
            n_total = 0
            n_skip = 0
            for row in iter_jsonl(train_path):
                n_total += 1
                sid = str(row["id"])
                if sid in done_ids:
                    n_skip += 1
                    continue
                ex = next(rr)
                futures.append(ex.submit(worker_save_one, dataset.name, row))

            tqdm_write(
                f"  - train total={n_total}, skip(done)={n_skip}, todo={len(futures)}"
            )

            if not futures:
                tqdm_write(f"  - nothing to do. already complete: {out_path}")
                continue

            ok_count = 0
            fail_count = 0

            pbar = tqdm(
                desc=f"save_memory {dataset.name}[train]",
                total=len(futures),
                unit="sample",
                dynamic_ncols=True,
            )
            pbar.set_postfix(ok=ok_count, fail=fail_count)

            for fut in as_completed(futures):
                try:
                    ok, mem_rec, err, pid = fut.result()
                except Exception:
                    fail_count += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=ok_count, fail=fail_count)
                    continue

                if not ok:
                    fail_count += 1
                    tqdm_write(
                        f"[save_memory][FAIL] {dataset.name} | SID={mem_rec['id']} | {err} | pid={pid}"
                    )
                    pbar.update(1)
                    pbar.set_postfix(ok=ok_count, fail=fail_count)
                    continue

                sid = str(mem_rec["id"])
                if sid in done_ids:
                    pbar.update(1)
                    pbar.set_postfix(ok=ok_count, fail=fail_count)
                    continue

                append_jsonl(out_path, mem_rec)
                done_ids.add(sid)

                for step_name in ("A1", "A2", "B1", "B2"):
                    step = mem_rec[step_name]
                    log_agent(sid, step_name, step["prompt"], step["output"])

                ok_count += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok_count, fail=fail_count)

            pbar.close()
            tqdm_write(f"  - updated memory jsonl: {out_path}")

    finally:
        for ex in executors:
            ex.shutdown(wait=True, cancel_futures=False)
