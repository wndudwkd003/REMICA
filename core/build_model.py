# core/build_model.py
from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model  # LoRA용
from transformers import AutoModelForCausalLM  # LLM용

from config.config import Config
from core.text_models import (
    BigBirdRoberta_MLP,
    DeBERTaV3_MLP,
    Longformer_MLP,
    ModernBERT_MLP,
    Roberta_MLP,
)


def build_model(
    config: Config,
    *,
    hidden_dim: int | None = None,
    num_labels: int = 2,
    llm: bool = False,
) -> nn.Module:
    model_id = (config.model_id or "").strip()
    mid = model_id.lower()

    # --------------------------------------------------
    # 1) LLM + LoRA 분기 (여기에 16비트 설정)
    # --------------------------------------------------
    if llm:
        llm_model_id = model_id

        # 1-1) dtype 결정: bf16 가능하면 bf16, 아니면 fp16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        # 1-2) LLM 로드할 때 16비트로 로딩
        base_model = AutoModelForCausalLM.from_pretrained(
            llm_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,  # 선택이지만 메모리 절약에 도움
        )

        # 1-3) LoRA 적용 (Config.use_lora 가 True일 때만)
        if config.use_lora:
            assert config.lora_target_modules, "lora_target_modules가 비어 있습니다."

            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias=config.lora_bias,          # "none" | "all" | "lora_only"
                task_type=TaskType.CAUSAL_LM,
                target_modules=config.lora_target_modules,
            )
            model = get_peft_model(base_model, lora_config)
        else:
            model = base_model

        return model

    # --------------------------------------------------
    # 2) CLS 분기 (기존 ModernBERT/DeBERTa 등)
    # --------------------------------------------------
    hd = hidden_dim if hidden_dim is not None else config.hidden_dim

    if "modernbert" in mid:
        return ModernBERT_MLP(
            model_id=model_id,
            hidden_dim=hd,
            num_labels=num_labels,
            trust_remote_code=True,
        )

    if "roberta" in mid and "bigbird" not in mid:
        return Roberta_MLP(
            model_id=model_id,
            hidden_dim=hd,
            num_labels=num_labels,
            trust_remote_code=False,
        )

    if "deberta" in mid:
        return DeBERTaV3_MLP(
            model_id=model_id,
            hidden_dim=hd,
            num_labels=num_labels,
            trust_remote_code=False,
        )

    if "longformer" in mid:
        return Longformer_MLP(
            model_id=model_id,
            hidden_dim=hd,
            num_labels=num_labels,
            trust_remote_code=False,
        )

    if "bigbird" in mid:
        return BigBirdRoberta_MLP(
            model_id=model_id,
            hidden_dim=hd,
            num_labels=num_labels,
            trust_remote_code=False,
        )

    raise ValueError(
        f"Unknown model_id={config.model_id!r}. "
        f"Supported encoders: answerdotai/ModernBERT-base | FacebookAI/roberta-base | "
        f"microsoft/deberta-v3-base | allenai/longformer-base-4096 | google/bigbird-roberta-base "
        f"or use llm=True for LLMs."
    )
