# utils/gpt_client.py
from __future__ import annotations

import json
import time
from typing import Any, Dict

from openai import OpenAI
from pydantic import BaseModel, Field


class RemStage1Out(BaseModel):
    pred_label: int = Field(..., description="0=appropriate, 1=inappropriate")
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str


class GPTClient:
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
        # Structured parse (성공하면 가장 깔끔)
        resp = self.client.responses.parse(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=self.max_output_tokens,
            text_format=RemStage1Out,
        )
        out = resp.output_parsed
        if out is None:
            raise RuntimeError("responses.parse returned output_parsed=None")
        return out.model_dump()

    def _call_raw_and_parse(self, prompt: str) -> Dict[str, Any]:
        # Raw create -> output_text -> json.loads
        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=self.max_output_tokens,
        )
        raw_text = (getattr(resp, "output_text", None) or "").strip()
        if not raw_text:
            # fallback: stringify
            raw_text = str(resp).strip()

        # 코드펜스 제거(혹시라도)
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`").strip()
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()

        data = json.loads(raw_text)
        checked = RemStage1Out.model_validate(data)
        return checked.model_dump()

    def call_api(self, prompt: str) -> Dict[str, Any]:
        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # 1) parse 먼저 시도
                return self._call_parse(prompt)

            except Exception as e1:
                # parse는 ValidationError로 죽을 수 있음 -> raw fallback
                last_err = e1
                try:
                    return self._call_raw_and_parse(prompt)
                except Exception as e2:
                    last_err = e2

            if attempt < self.max_retries:
                time.sleep(self.retry_sleep)
                continue

            raise last_err
