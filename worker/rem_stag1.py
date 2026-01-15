# worker/rem_stag1.py

from utils.data_utils import JsonlDataset
from utils.prompt_utils import build_rem_stage1_prompt
import json
from utils.gpt_client import GPTClient
from config.config import REM_STEP_1_DATASET, Config
from pathlib import Path
from utils.db_utils import open_db, init_schema, exists, upsert


def run_rem_stage1(config: Config):
    split = config.rem_split

    print(f"[rem_stage1.py] split: {split}")

    rem_dir = Path(config.rem_dir)

    db_path = rem_dir / f"stage_1.sqlite3"

    client = GPTClient(
        model=config.gpt_model,
        temperature=config.gpt_temperature,
        max_output_tokens=config.gpt_max_output_tokens,
    )

    conn = open_db(db_path)
    init_schema(conn)

    for ds in REM_STEP_1_DATASET:
        ds_path = Path(config.datasets_dir) / ds.value / f"{split}.jsonl"
        print(f"[rem_stage1.py] Processing dataset: {ds.value}, path: {ds_path}")

        dataset = JsonlDataset(ds_path, meta_to_text=False)

        for i in range(len(dataset)):
            sid, text, label, metadata = dataset[i]

            if exists(conn, sid):
                continue

            prompt = build_rem_stage1_prompt(text, ds)

            out = client.call_api(prompt)

            pred_label = out["pred_label"]
            confidence = out["confidence"]
            rationale = out["rationale"]

            upsert(
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
