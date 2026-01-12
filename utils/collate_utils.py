import torch
from transformers import AutoTokenizer, DebertaV2Tokenizer
from config.config import FAST_NOT_MODEL


class TextCollator:
    def __init__(self, model_id: str, max_len: int, trust_remote_code: bool):
        if model_id in FAST_NOT_MODEL:
            self.tok = DebertaV2Tokenizer.from_pretrained(model_id)
        else:
            self.tok = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=True,
                trust_remote_code=True,
            )
        self.max_len = max_len

    def __call__(self, batch):
        sids, texts, labels, metas = zip(*batch)

        model_max = int(getattr(self.tok, "model_max_length", self.max_len))
        if model_max > 1000000:
            model_max = self.max_len
        max_len = int(min(self.max_len, model_max))

        enc = self.tok(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        return {
            "ids": list(sids),
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
            "metadata": list(metas),
        }
