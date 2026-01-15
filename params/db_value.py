# params/db_value.py


from enum import Enum


class DB(Enum):
    REM_STAGE_1 = "rem_stage_1"
    REM_STAGE_2 = "rem_stage_2"

    ID = "id"
    PRED_LABEL = "pred_label"
    CONFIDENCE = "confidence"
    RATIONALE = "rationale"
    RAW_JSON = "raw_json"
    CREATED_AT = "created_at"

    # stage2
    GOLD_LABEL = "gold_label"
    VERDICT = "verdict"
    FINAL_LABEL = "final_label"
    SUPPORT_EVIDENCE = "support_evidence"
    ERROR_EVIDENCE = "error_evidence"
    MISSING_EVIDENCE = "missing_evidence"
    MEMORY = "memory"
