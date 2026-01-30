# run.py

from __future__ import annotations

from config.config import Config, DoModeEnum
from utils.api_utils import token_key_regist
from utils.seeds_utils import set_seeds

from worker.save_memory import run_save_memory
from worker.test_memory import run_test_memory

# from worker.debate_direct import (
#     run_debate_direct_test,
#     run_nothing_direct_test,
# )

from worker.model_train import run_model_train, run_model_test


def main(config: Config):
    if config.do_mode == DoModeEnum.SAVE_DEBATE_MEMORY:
        run_save_memory(config)

    elif config.do_mode == DoModeEnum.TEST_DEBATE_MEMORY:
        run_test_memory(config)


if __name__ == "__main__":
    config = Config()
    print(f"Current api model: {config.api_model.value}")

    # API 키 등록
    token_key_regist(config.api_json_path)
    # 시드 고정
    set_seeds(config.seed)
    # 메인 실행
    main(config)

    print("=== All Process Finished ===")
