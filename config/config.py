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


class ContextDatasetEnum(Enum):
    PROSOCIAL_DIALOG = "prosocial-dialog"
    BOT_ADVERSARIAL_DIALOGUE = "bot-adversarial-dialogue"
    TOXICHAT = "toxichat"


REM_STEP_1_DATASET = [
    DatasetEnum.HSOL,
    DatasetEnum.RealToxicityPrompts,
    DatasetEnum.ToxiSpanSE,
    DatasetEnum.HSDCD,
]

DATASET_BS = 1
DATASET_ORDER = [
    (dataset, DATASET_BS) for dataset in REM_STEP_1_DATASET
]

ICA_STEP_DATASET = [
    ContextDatasetEnum.PROSOCIAL_DIALOG,
    ContextDatasetEnum.BOT_ADVERSARIAL_DIALOGUE,
    ContextDatasetEnum.TOXICHAT,
]


class ModelEnum(Enum):
    ModernBERT = "answerdotai/ModernBERT-base"
    RoBERTa = "FacebookAI/roberta-base"
    DeBERTaV3 = "microsoft/deberta-v3-base"
    Longformer = "allenai/longformer-base-4096"
    BigBird = "google/bigbird-roberta-base"

    GEMMA2_2 = "google/gemma-2-2b-it"
    LLAMA3_2 = "meta-llama/Llama-3.2-3B-Instruct"
    PHI3_5 = "microsoft/Phi-3.5-mini-instruct"


FAST_NOT_MODEL = [ModelEnum.DeBERTaV3.value]


# Select the model to use for experiments!!!!!!!!!!!!!!!!
SELECT_MODEL = ModelEnum.GEMMA2_2


@dataclass
class Config:
    use_llm_classifier: bool = True  # True | False

    llm_target_mode: str = "rem12"  # "rem12" | "simple" 출력 내용 설정

    use_rem2_aug: bool = True  # True | False 유사사례 증강할지말지
    rem2_top_k: int = 3


    rem2_faiss_index_path: str = "rem/faiss/rem2_text_train.index"
    rem2_faiss_meta_path: str = "rem/faiss/rem2_text_train.meta.jsonl"

    emb_device: str = "cuda"  # "cuda" or "cpu"
    rem_gpus: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    datasets_dir: str = "datasets_processed"
    context_datasets_dir: str = "datasets_context_processed"

    dataset_sum: bool = False  # True | False
    dataset_sum_batch_size: int = 1
    dataset_order: list[tuple[DatasetEnum, int]] = field(
        default_factory=lambda: DATASET_ORDER
    )
    # config.Config 안에

    rem_mode: str = "rem12" # simple | rem1 | rem12 유사사례 어떻게

    do_mode: str | None = (
        None  # REM_Stage_1 | REM_Stage_2 | ICA | GPT_INFER | check | None
    )
    api_json_path: str = "config/api.json"
    rem_step1_datasets: list[DatasetEnum] = field(
        default_factory=lambda: REM_STEP_1_DATASET
    )
    ica_step_datasets: list[ContextDatasetEnum] = field(
        default_factory=lambda: ICA_STEP_DATASET
    )
    rem_split: str = "train"
    remica_db_path: str = "rem/remica.sqlite3"
    rem_worker: int = 12

    # GPT 관련
    gpt_model: str = (
        "gpt-4.1-2025-04-14"  # "gpt-5-mini-2025-08-07" # gpt-5.2-2025-12-11 # gpt-5-mini-2025-08-07 # gpt-4.1-2025-04-14
    )
    gpt_temperature: float = 0.0
    gpt_top_p: float = 1.0
    gpt_max_output_tokens: int = 2048
    gpt_concurrency: int = 8
    max_retries: int = 5

    model_name: str = SELECT_MODEL.name
    model_id: str = SELECT_MODEL.value

    run_dir: str = "runs"
    train_mode: str = "train_test"  # "train" or "test"
    load_run_dir: str | None = None  # test에서 특정 run 폴더를 지정하고 싶으면 사용

    seed: int = 42 # 42 2026 1234

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

    rem_1_top_k: int = 3
    emb_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    emb_batch_size: int = 128
    sim_index_dir: str = "rem/sim_index"


    # --- LoRA 관련 ---
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_bias: str = "all"
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "v_proj", "k_proj", "o_proj", "out_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    )
