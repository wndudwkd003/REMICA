# utils/gpt_client.py

import json
import time
from typing import Any, Dict

from openai import OpenAI
from pydantic import BaseModel, Field


class RemStage1Out(BaseModel):
    pred_label: int = Field(..., description="0=appropriate, 1=inappropriate")
    rationale: str = Field(..., description="설명")


def _extract_json_obj(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty_output_text")

    # 코드펜스 제거
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    # 앞뒤 잡텍스트가 섞여도 JSON 객체만 슬라이스
    l = raw.find("{")
    r = raw.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError(f"no_json_object: {raw[:120]}")
    return raw[l : r + 1]


class GPTClient:
    def __init__(
        self,
        model: str,
        max_output_tokens: int,
        max_retries: int = 3,
        retry_sleep: float = 0.5,
        top_p: float = 1.0,
        temperature: float = 0.0,
    ):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.top_p = top_p
        self.temperature = temperature
        self.client = OpenAI()

    def _call_parse(self, prompt: str) -> Dict[str, Any]:
        resp = self.client.responses.parse(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=self.max_output_tokens,
            text_format=RemStage1Out,
            temperature=self.temperature,
            top_p=self.top_p,
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

        raw_text = getattr(resp, "output_text", None) or ""
        json_str = _extract_json_obj(raw_text)

        data = json.loads(json_str)
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
