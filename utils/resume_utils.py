# utils/resume_utils.py

import json
import shutil
from datetime import datetime
from enum import Enum
from pathlib import Path


def load_existing_results(run_dir: Path) -> dict:
    """
    기존 cross_test.json이 있으면 로드해서 이어쓰기.
    없으면 빈 dict 반환.
    """
    path = run_dir / "results" / "cross_test.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def backup_partial_save_dir_if_exists(run_dir: Path, ds):
    """
    완료가 아닌데(save/<ds>) 폴더가 이미 존재하면, 덮어쓰기 방지를 위해 백업 폴더로 이동.
    DatasetEnum 또는 문자열 이름 둘 다 지원.
    """
    name = _ds_name(ds)
    save_dir = run_dir / "save" / name
    if not save_dir.exists():
        return

    # 이미 완료(B안 기준)면 백업할 필요 없음
    if is_done_dataset(run_dir, ds):
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = save_dir.parent / f"{name}__partial_backup_{ts}"
    shutil.move(str(save_dir), str(backup_dir))
    print(f"[RESUME-B] moved partial save_dir -> {backup_dir}")


def _ds_name(ds: Enum | str) -> str:
    """
    DatasetEnum 또는 str 둘 다 받아서 name을 문자열로 반환.
    """
    return ds.name if hasattr(ds, "name") else str(ds)


def is_done_dataset(run_dir: Path, ds) -> bool:
    name = _ds_name(ds)
    save_dir = run_dir / "save" / name
    required = [
        "best.pt",
        "meta.json",
        "train_valid_history.csv",
        "train_valid_loss.png",
        "train_valid_acc.png",
    ]
    return all((save_dir / fn).exists() for fn in required)
