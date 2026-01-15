# worker/rem_stag2.py
from __future__ import annotations

import json
from pathlib import Path

from config.config import Config, REM_STEP_1_DATASET
from utils.data_utils import JsonlDataset
from utils.retriever import SimilarTextRetriever
from utils.prompt_utils import build_rem_stage2_prompt
from utils.db_utils import open_db, init_schema, exists, get_stage1, upsert_stage2
from params.db_value import DB
from utils.gpt_client_stage2 import GPTClientStage2


def run_rem_stage2(config: Config):
    split = config.rem_split
    print(f"[rem_stage2.py] split: {split}")

    rem_dir = Path(config.rem_dir)
    db1_path = rem_dir / "stage_1.sqlite3"
    db2_path = rem_dir / "stage_2.sqlite3"

    client = GPTClientStage2(
        model=config.gpt_model,
        max_output_tokens=config.gpt_max_output_tokens,
    )

    retriever = SimilarTextRetriever(config)

    conn1 = open_db(db1_path)
    init_schema(conn1)

    conn2 = open_db(db2_path)
    init_schema(conn2)

    for ds in REM_STEP_1_DATASET:
        ds_name = ds.name
        ds_path = Path(config.datasets_dir) / ds_name / f"{split}.jsonl"
        print(f"[rem_stage2.py] Processing dataset: {ds_name}, path: {ds_path}")

        dataset = JsonlDataset(ds_path, meta_to_text=False)

        for i in range(len(dataset)):
            sid, text, gold_label, metadata = dataset[i]
            gold_label = int(gold_label)

            # stage2 이미 있으면 skip
            if exists(conn2, DB.REM_STAGE_2.value, sid):
                continue

            # stage1 결과 없으면 skip (stage1에서 실패/미저장)
            s1 = get_stage1(conn1, sid)
            if s1 is None:
                continue

            stage1_pred = int(s1["pred_label"])
            stage1_rationale = s1["rationale"]

            verdict_ok = stage1_pred == gold_label
            label_for_neighbors = gold_label if verdict_ok else stage1_pred

            sim_texts = retriever.get_similar_texts(
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
                gold_label=gold_label,
                stage1_pred=stage1_pred,
                stage1_rationale=stage1_rationale,
            )

            try:
                out = client.call_api(prompt)
            except Exception as e:
                print(f"[rem_stage2.py] FAIL sid={sid} err={e}")
                continue

            upsert_stage2(
                conn2,
                sid=sid,
                gold_label=gold_label,
                verdict=out["verdict"],
                final_label=int(out["final_label"]),
                support_evidence=out["support_evidence"],
                error_evidence=out["error_evidence"],
                missing_evidence=out["missing_evidence"],
                memory=out["memory"],
                raw_json=json.dumps(out, ensure_ascii=False),
            )

        conn2.commit()
        print(f"[rem_stage2.py] Completed dataset: {ds_name}")

    conn1.close()
    conn2.close()
    print(f"[rem_stage2.py] DONE. DB saved at: {db2_path}")
    return db2_path
