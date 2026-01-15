# params/db_value.py


from enum import Enum


class DB(Enum):
    REM_STAGE_1 = "rem_stage_1"
    ID = "id"
    PRED_LABEL = "pred_label"
    CONFIDENCE = "confidence"
    RATIONALE = "rationale"
    RAW_JSON = "raw_json"
    CREATED_AT = "created_at"
