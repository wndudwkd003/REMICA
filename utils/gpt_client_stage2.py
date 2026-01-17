# utils/gpt_client_stage2.py
from __future__ import annotations

import json
import time
from typing import Any, Dict

from openai import OpenAI
from pydantic import BaseModel, Field


class RemStage2Out(BaseModel):
    evidence: str = Field(
        ...,
        description="single evidence string; correct=key support, wrong='ERROR: ... | MISSING: ...'",
    )
    memory: str = Field(..., description="one-line reusable rule; no quotes")


class GPTClientStage2:
    def __init__(
        self,
        model: str,
        max_output_tokens: int,
        max_retries: int = 3,
        retry_sleep: float = 0.5,
    ):
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.max_retries = int(max_retries)
        self.retry_sleep = float(retry_sleep)
        self.client = OpenAI()

    def _call_parse(self, prompt: str) -> Dict[str, Any]:
        resp = self.client.responses.parse(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=self.max_output_tokens,
            text_format=RemStage2Out,
        )
        out = resp.output_parsed
        if out is None:
            raise RuntimeError("responses.parse returned output_parsed=None")
        return out.model_dump()

    def _call_raw_and_parse(self, prompt: str) -> Dict[str, Any]:
        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=self.max_output_tokens,
        )
        raw_text = (getattr(resp, "output_text", None) or "").strip()
        if not raw_text:
            raw_text = str(resp).strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").strip()
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()

        data = json.loads(raw_text)
        checked = RemStage2Out.model_validate(data)
        return checked.model_dump()

    def call_api(self, prompt: str) -> Dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_parse(prompt)
            except Exception as e1:
                last_err = e1
                try:
                    return self._call_raw_and_parse(prompt)
                except Exception as e2:
                    last_err = e2

            if attempt < self.max_retries:
                time.sleep(self.retry_sleep)
                continue
            raise last_err
