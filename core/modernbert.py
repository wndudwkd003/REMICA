# core/modernbert.py

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
