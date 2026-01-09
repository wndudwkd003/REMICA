# core/text_models.py

import torch
import torch.nn as nn
from transformers import AutoModel


class ModernBERT_MLP(nn.Module):
    def __init__(
        self,
        model_id: str,
        hidden_dim: int = 512,
        num_labels: int = 2,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        self.enc = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        h = self.enc.config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(h, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.enc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0]  # [B, H]
        logits = self.mlp(cls)  # [B, 2]
        return logits


class Roberta_MLP(nn.Module):
    def __init__(
        self,
        model_id: str = "FacebookAI/roberta-base",
        hidden_dim: int = 512,
        num_labels: int = 2,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        self.enc = AutoModel.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        h = self.enc.config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(h, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.enc(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        cls = out.last_hidden_state[:, 0]  # RoBERTa: <s> token
        logits = self.mlp(cls)
        return logits


class DeBERTaV3_MLP(nn.Module):
    def __init__(
        self,
        model_id: str = "microsoft/deberta-v3-base",
        hidden_dim: int = 512,
        num_labels: int = 2,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        self.enc = AutoModel.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        h = self.enc.config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(h, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.enc(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        cls = out.last_hidden_state[:, 0]  # DeBERTa: [CLS] 위치
        logits = self.mlp(cls)
        return logits


class Longformer_MLP(nn.Module):
    """
    Longformer는 분류 시 보통 첫 토큰을 global attention으로 설정합니다.
    global_attention_mask가 None이면 자동으로 (B, L) 만들고 첫 토큰만 1로 둡니다.
    """

    def __init__(
        self,
        model_id: str = "allenai/longformer-base-4096",
        hidden_dim: int = 512,
        num_labels: int = 2,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        self.enc = AutoModel.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        h = self.enc.config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(h, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask, global_attention_mask=None):
        if global_attention_mask is None:
            # Longformer 권장 세팅: 첫 토큰(global) = 1
            global_attention_mask = torch.zeros_like(attention_mask)
            global_attention_mask[:, 0] = 1

        out = self.enc(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            return_dict=True,
        )
        cls = out.last_hidden_state[:, 0]
        logits = self.mlp(cls)
        return logits


class BigBirdRoberta_MLP(nn.Module):
    def __init__(
        self,
        model_id: str = "google/bigbird-roberta-base",
        hidden_dim: int = 512,
        num_labels: int = 2,
        trust_remote_code: bool = False,
    ):
        super().__init__()
        self.enc = AutoModel.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        h = self.enc.config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(h, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        out = self.enc(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        cls = out.last_hidden_state[:, 0]  # RoBERTa 계열처럼 <s>
        logits = self.mlp(cls)
        return logits
