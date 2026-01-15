from dataclasses import dataclass, field
from enum import Enum


class DatasetEnum(Enum):
    DiaSafety = "DiaSafety"
    GabHate = "gab_hate"
    HSOL = "hate-speech-and-offensive-language-master"
    HateXplain = "hatexplain"
    RealToxicityPrompts = "real-toxicity-prompts"
    OffenseEval = "offenseval"
    HSD = "hate-speech-dataset-master"
    ToxiGen = "toxigen"
    ToxiSpanSE = "ToxiSpanSE"
    ToxiCR = "toxicr"
    HSDCD = "hsdcd"
    ISHate = "ISHate"


REM_STEP_1_DATASET = [
    DatasetEnum.HSOL,
    DatasetEnum.HateXplain,
    DatasetEnum.DiaSafety,
    DatasetEnum.ToxiSpanSE,
    DatasetEnum.HSD,
]


class ModelEnum(Enum):
    ModernBERT = "answerdotai/ModernBERT-base"
    RoBERTa = "FacebookAI/roberta-base"
    DeBERTaV3 = "microsoft/deberta-v3-base"
    Longformer = "allenai/longformer-base-4096"
    BigBird = "google/bigbird-roberta-base"


FAST_NOT_MODEL = [ModelEnum.DeBERTaV3.value]


# Select the model to use for experiments!!!!!!!!!!!!!!!!
SELECT_MODEL = ModelEnum.Longformer


@dataclass
class Config:
    datasets_dir: str = "datasets_processed"

    dataset_order: list[tuple[DatasetEnum, int]] = field(
        default_factory=lambda: [
            (DatasetEnum.DiaSafety, 4),
            (DatasetEnum.GabHate, 4),
            (DatasetEnum.HateXplain, 2),
            (DatasetEnum.HSOL, 2),
            (DatasetEnum.RealToxicityPrompts, 2),
            (DatasetEnum.OffenseEval, 2),
            (DatasetEnum.HSD, 2),
            (DatasetEnum.ToxiGen, 2),
            (DatasetEnum.ToxiSpanSE, 2),
            (DatasetEnum.ToxiCR, 2),
            (DatasetEnum.HSDCD, 2),
            (DatasetEnum.ISHate, 2),
        ]
    )

    do_mode: str = "REM_Stage_1"  # REM_Stage_1 | REM_Stage_2 |
    api_json_path: str = "config/api.json"
    rem_step1_datasets: list[DatasetEnum] = field(
        default_factory=lambda: REM_STEP_1_DATASET
    )
    rem_split: str = "train"
    rem_dir: str = "rem"

    # GPT 관련
    gpt_model: str = "gpt-5-mini-2025-08-07"
    gpt_temperature: float = 0.0
    gpt_max_output_tokens: int = 512
    gpt_concurrency: int = 8
    max_retries: int = 5

    model_name: str = SELECT_MODEL.name
    model_id: str = SELECT_MODEL.value

    run_dir: str = "runs"
    train_mode: str = "train_test"  # "train" or "test"
    load_run_dir: str | None = None  # test에서 특정 run 폴더를 지정하고 싶으면 사용

    seed: int = 42
    max_len: int = 2048
    num_epochs: int = 200
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_workers: int = 0

    meta_to_text: bool = False
    hidden_dim: int = 512

    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001

    sim_k: int = 3
    emb_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    emb_batch_size: int = 128
    sim_index_dir: str = "sim_index"
