# utils/collate_utils.py

import torch
from transformers import AutoTokenizer, DebertaV2Tokenizer

from config.config import FAST_NOT_MODEL


def build_hf_tokenizer(model_id: str, trust_remote_code: bool = True):
    """
    CLS/LLM 공통 HuggingFace 토크나이저 빌더.

    - DeBERTa 계열은 DebertaV2Tokenizer
    - 그 외는 AutoTokenizer(use_fast=True, trust_remote_code=...)
    - pad_token 없으면 eos_token으로 채움 (LLM 학습 시 필수)
    """
    if model_id in FAST_NOT_MODEL:
        tok = DebertaV2Tokenizer.from_pretrained(model_id)
    else:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )

    # pad_token 없으면 eos_token으로 맞춰줌
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token

    return tok


class TextCollator:
    def __init__(self, model_id: str, max_len: int, trust_remote_code: bool):
        self.tok = build_hf_tokenizer(model_id, trust_remote_code=trust_remote_code)
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


class LMCollator:
    """
    Rem2LMDataset 출력(batch: dict들)을 받아
    prompt + target 을 이어붙여 causal LM 학습용 input_ids / labels 생성.
    """

    def __init__(
        self,
        model_id: str,
        max_len: int,
        trust_remote_code: bool = True,
        ignore_index: int = -100,
    ):
        # 여기서 직접 HF 토크나이저 생성
        self.tokenizer = build_hf_tokenizer(
            model_id=model_id,
            trust_remote_code=trust_remote_code,
        )
        self.max_len = max_len
        self.ignore_index = ignore_index

    def __call__(self, batch):
        prompts = [b["prompt"] for b in batch]
        targets = [b["target"] for b in batch]

        full_texts = [p + "\n\n" + t for p, t in zip(prompts, targets)]

        enc = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]

        labels = input_ids.clone()

        # prompt 부분은 loss를 안 보게 ignore_index로 마스킹
        for i, p in enumerate(prompts):
            p_ids = self.tokenizer(
                p + "\n\n",
                add_special_tokens=False,
            )["input_ids"]
            plen = min(len(p_ids), self.max_len)
            labels[i, :plen] = self.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }
