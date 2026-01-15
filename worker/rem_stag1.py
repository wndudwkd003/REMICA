# worker/rem_stag1.py

from utils.data_utils import JsonlDataset
from utils.prompt_utils import build_rem_stage1_prompt
import json
from utils.gpt_client_stage1 import GPTClient
from config.config import REM_STEP_1_DATASET, Config
from pathlib import Path
from utils.db_utils import open_db, init_schema, exists, upsert_stage1

from utils.retriever import SimilarTextRetriever
from params.db_value import DB


def run_rem_stage1(config: Config):
    split = config.rem_split

    print(f"[rem_stage1.py] split: {split}")

    rem_dir = Path(config.rem_dir)

    db_path = rem_dir / f"stage_1.sqlite3"

    client = GPTClient(
        model=config.gpt_model,
        max_output_tokens=config.gpt_max_output_tokens,
        max_retries=config.max_retries,
    )

    retriever = SimilarTextRetriever(config)

    conn = open_db(db_path)
    init_schema(conn)

    for ds in REM_STEP_1_DATASET:
        ds_name = ds.name
        ds_path = Path(config.datasets_dir) / ds_name / f"{split}.jsonl"
        print(f"[rem_stage1.py] Processing dataset: {ds_name}, path: {ds_path}")

        dataset = JsonlDataset(ds_path, meta_to_text=False)

        for i in range(len(dataset)):
            sid, text, label, metadata = dataset[i]

            if exists(conn, DB.REM_STAGE_1.value, sid):
                continue

            sim_texts = retriever.get_similar_texts(
                ds=ds,
                label=label,
                query_sid=sid,
                query_text=text,
                top_k=config.sim_k,
            )

            prompt = build_rem_stage1_prompt(text, ds, sim_texts)

            out = client.call_api(prompt)

            pred_label = out["pred_label"]
            confidence = out["confidence"]
            rationale = out["rationale"]

            print(
                f"[rem_stage1.py] {ds.value} | SID: {sid} | True: {label} | Pred: {pred_label} | Conf: {confidence} | Rationale: {rationale}"
            )

            upsert_stage1(
                conn,
                sid,
                pred_label,
                confidence,
                rationale,
                json.dumps(out, ensure_ascii=False),
            )

        conn.commit()
        print(f"[rem_stage1.py] Completed dataset: {ds.value}")

    conn.close()
    print(f"[rem_stage1.py] All datasets processed. DB saved at: {db_path}")
    return db_path
