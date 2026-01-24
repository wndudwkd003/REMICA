# run.py
import json
import os

from config.config import Config
from utils.db_utils import open_db
from utils.seeds_utils import set_seeds
from worker.gpt_infer import run_gpt_infer
from worker.ica_stage import run_ica
from worker.rem_stage1 import run_rem_stage1
from worker.rem_stage2 import run_rem_stage2
from worker.trainer import test, train


def token_key_regist(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, value in data.items():
        os.environ[key] = value
        print(f"Registered token for {key}")


def _db_check(config: Config):
    db_path = config.remica_db_path
    conn = open_db(db_path)

    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
    ]
    print(f"[CHECK] DB: {db_path}")
    print(f"[CHECK] tables: {tables}")

    for t in tables:
        cols = conn.execute(f"PRAGMA table_info({t});").fetchall()
        col_names = [c[1] for c in cols]

        cnt = conn.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]

        print("\n" + "=" * 80)
        print(f"[TABLE] {t}")
        print(f"  columns({len(col_names)}): {col_names}")
        print(f"  rows: {cnt}")

        if cnt > 0:
            sample = conn.execute(f"SELECT * FROM {t} LIMIT 1;").fetchone()
            sample_dict = {col_names[i]: sample[i] for i in range(len(col_names))}
            print("  sample_row:")
            print(json.dumps(sample_dict, ensure_ascii=False, indent=2)[:2000])

    conn.close()




def main(config: Config):
    print(f"Current Model: {config.model_name} ({config.model_id})")
    print(f"do_mode: {config.do_mode}")

    if config.do_mode == "check":
        _db_check(config)
        return

    if config.do_mode == "REM_Stage_1":
        out_db = run_rem_stage1(config)
        print(f"[DONE] REM Stage1 db: {out_db}")
        return
    elif config.do_mode == "REM_Stage_2":
        out_db = run_rem_stage2(config)
        print(f"[DONE] REM Stage2 db: {out_db}")
        return
    elif config.do_mode == "ICA":
        out_db = run_ica(config)
        print(f"[DONE] ICA db: {out_db}")
        return

    elif config.do_mode == "GPT_INFER":

        out_path = run_gpt_infer(config)  # 여기서 예측 결과 저장까지
        print(f"[DONE] GPT_INFER output: {out_path}")
        return


    if config.train_mode == "train":
        run_dir = train(config)
        print(f"[DONE] train run_dir: {run_dir}")

    elif config.train_mode == "train_test":
        run_dir = train(config)
        print(f"[DONE] train run_dir: {run_dir}")

        config.load_run_dir = run_dir
        run_dir = test(config)
        print(f"[DONE] test run_dir: {run_dir}")

    else:
        run_dir = test(config)
        print(f"[DONE] test run_dir: {run_dir}")


if __name__ == "__main__":
    config = Config()
    token_key_regist(config.api_json_path)
    set_seeds(config.seed)
    main(config)
