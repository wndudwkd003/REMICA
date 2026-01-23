# params/db_value.py
from enum import Enum


class DB(Enum):
    # tables
    REM_STAGE_1 = "rem_stage_1"
    REM_STAGE_2 = "rem_stage_2"
    ICA = "ica"

    # shared
    ID = "id"
    RAW_JSON = "raw_json"
    CREATED_AT = "created_at"

    # stage1 cols
    PRED_LABEL = "pred_label"
    RATIONALE = "rationale"

    # stage2 cols
    TRUE_LABEL = "true_label"
    IS_CORRECT = "is_correct"
    EVIDENCE = "evidence"
    MEMORY = "memory"
    RELIABILITY = "reliability"
    RUNS_JSON = "runs_json"

    # ica cols (핵심 메타)
    SOURCE_DATASET = "source_dataset"
    SPLIT = "split"

    # ---- 원본 샘플 역추적 포인터 ----
    SOURCE_CID = "source_cid"
    SOURCE_FILE = "source_file"
    WINDOW_START = "window_start"
    WINDOW_END = "window_end"
    LAST_TEXT = "last_text"

    # ---- ICA 추출 결과 ----
    CONVERSATION_LABEL = "conversation_label"
    CONTEXT_SUMMARY = "context_summary"
    TRIGGERS_JSON = "triggers_json"
    TARGETS_JSON = "targets_json"
    RULES_JSON = "rules_json"
