# core/build_model.py
from __future__ import annotations

import torch.nn as nn

from config.config import Config
from core.text_models import (
    ModernBERT_MLP,
    Roberta_MLP,
    DeBERTaV3_MLP,
    Longformer_MLP,
    BigBirdRoberta_MLP,
)


def build_model(
    config: Config, *, hidden_dim: int | None = None, num_labels: int = 2
) -> nn.Module:
    hd = hidden_dim if hidden_dim is not None else config.hidden_dim

    model_id = (config.model_id or "").strip()
    mid = model_id.lower()

    # ---- model_id 기반 분기 (model_name은 무시) ----
    if "modernbert" in mid:
        return ModernBERT_MLP(
            model_id=model_id,
            hidden_dim=hd,
            num_labels=num_labels,
            trust_remote_code=True,
        )

    # roberta-base (단, bigbird-roberta는 아래에서 처리)
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
        f"Supported: answerdotai/ModernBERT-base | FacebookAI/roberta-base | "
        f"microsoft/deberta-v3-base | allenai/longformer-base-4096 | google/bigbird-roberta-base"
    )
