# config/config.py

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


TARGET_DATASETS = [
    # DatasetEnum.HSOL,
    DatasetEnum.RealToxicityPrompts,
    DatasetEnum.ToxiSpanSE,
    DatasetEnum.HSDCD,
    # DatasetEnum.OffenseEval,
    DatasetEnum.DiaSafety,
    # DatasetEnum.HateXplain,
]


DATASET_BS = 4
DATASET_ORDER = [(dataset, DATASET_BS) for dataset in TARGET_DATASETS]


class ModelEnum(Enum):
    GPT5_MINI = "gpt-5-mini-2025-08-07"
    GPT5_1 = "gpt-5.1-2025-11-13"
    GPT4_1 = "gpt-4.1-2025-04-14"
    CLAUDE_SONNET4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_HAIKU4_5 = "claude-haiku-4-5-20251001"


class DoModeEnum(Enum):
    SAVE_DEBATE_MEMORY = "save_debate_memory"
    TEST_DEBATE_MEMORY = "test_debate_memory"
    ONLINE_TEST = "online_test"
    MODEL_TRAIN = "model_train"
    MODEL_TEST = "model_test"


class ChainEnum(Enum):
    CHAIN_OF_EXPERT = "chain_of_expert"
    CHAIN_OF_DEBATE = "chain_of_debate"
    NONE = "none"


@dataclass
class Config:
    is_sim_legacy: bool = True
    actual_label_intervention: bool = False
    chain_mode: ChainEnum = ChainEnum.CHAIN_OF_DEBATE
    do_mode: DoModeEnum = DoModeEnum.ONLINE_TEST
    memory_dir: str = "debate_memory"
    able_gpus: list[int] = field(default_factory=lambda: [0,1,2,3])
    datasets_dir: str = "datasets_processed"
    dataset_order: list[tuple[DatasetEnum, int]] = field(
        default_factory=lambda: DATASET_ORDER
    )
    api_json_path: str = "config/api.json"

    # GPT 관련
    multi_worker: int = 20
    api_model: ModelEnum = ModelEnum.GPT5_1
    temperature: float = 0.0
    top_p: float = 1.0
    max_retries: int = 5
    max_new_tokens: dict = field(
        default_factory=lambda: {
            "A1": 2048,
            "B1": 2048,
            "A2": 2048,
            "B2": 2048,
            "DIRECT": 4096,
            "INFER": 4096,
        }
    )

    run_dir: str = "runs"
    seed: int = 42  # 42 2026 1234
    max_len: int = 4096  # 2048
    num_epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    num_workers: int = 0
    meta_to_text: bool = False
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.001
    rag_top_k: int = 3
    rag_cand_k: int = 30
    rag_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_batch_size: int = 128
    rag_device: str = "cuda"
    faiss_dir: str = "faiss_index"




    tsne_perplexity: int = 30
    tsne_n_iter: int = 1500
    runs_analysis_dir: str = "runs_analysis"
    pca_dim_before_tsne: int = 50
