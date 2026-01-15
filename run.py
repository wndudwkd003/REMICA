# run.py
from utils.seeds_utils import set_seeds
from config.config import Config
from worker.trainer import train, test
from worker.rem_stag1 import run_rem_stage1
import json
import os


def token_key_regist(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key, value in data.items():
        # 환경 변수로 설정
        os.environ[key] = value
        print(f"Registered token for {key}")


def main(config: Config):
    print(f"Current Model: {config.model_name} ({config.model_id})")
    print(f"do_mode: {config.do_mode}")

    # REM Stage 1은 학습/테스트가 아니라 GPT 기반 데이터 생성 파이프라인
    if config.do_mode == "REM_Stage_1":
        out_db = run_rem_stage1(config)
        print(f"[DONE] REM Stage1 db: {out_db}")
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
