# utils/collate_utils.py


import torch
from transformers import AutoTokenizer


class TextCollator:
    def __init__(self, model_id: str, max_len: int, trust_remote_code: bool = True):
        self.tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        self.max_len = max_len

    def __call__(self, batch):
        # batch: (sid, text, label, metadata)
        sids, texts, labels, metas = zip(*batch)

        enc = self.tok(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "ids": list(sids),
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
            "metadata": list(metas),
        }
