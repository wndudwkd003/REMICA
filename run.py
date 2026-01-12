# run.py

from utils.seeds_utils import set_seeds
from config.config import Config

from worker.trainer import train, test


def main(config: Config):
    print(f"Current Model: {config.model_name} ({config.model_id})")

    if config.train_mode == "train":
        print("Training started...")

        run_dir = train(config)
        print(f"[DONE] train run_dir: {run_dir}")

    elif config.train_mode == "train_test":
        print("Training started...")

        run_dir = train(config)
        print(f"[DONE] train run_dir: {run_dir}")

        print("Testing started...")

        config.load_run_dir = run_dir

        run_dir = test(config)
        print(f"[DONE] test run_dir: {run_dir}")

    else:
        print("Test started...")

        run_dir = test(config)
        print(f"[DONE] test run_dir: {run_dir}")


if __name__ == "__main__":
    config = Config()
    set_seeds(config.seed)

    main(config)
