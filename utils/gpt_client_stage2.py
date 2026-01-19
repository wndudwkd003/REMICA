# utils/gpt_client_stage2.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Literal, Optional

from openai import OpenAI
from pydantic import BaseModel, Field


class RemStage2Out(BaseModel):
    pred_label: int = Field(..., description="0=appropriate, 1=inappropriate")
    evidence: str = Field(..., description="결정 근거(핵심 포인트만, 짧게)")
    memory: str = Field(
        ..., description="다음에 재사용 가능한 reflective memory(규칙/패턴/주의사항)"
    )

    run_tag: Literal["run0", "run1", "run2"] = Field(..., description="실행 라벨")
    used_rules: List[str] = Field(
        ...,
        description="이번 판단에 실제로 사용한 규칙(선택된 subset). 없으면 빈 리스트",
    )
    label_stable: Optional[bool] = Field(
        None, description="(선택) run0/1/2 비교 후 안정적이면 True"
    )


def _extract_json_obj(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty_output_text")

    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    l = raw.find("{")
    r = raw.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError(f"no_json_object: {raw[:120]}")
    return raw[l : r + 1]


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
        raw_text = getattr(resp, "output_text", None) or ""
        json_str = _extract_json_obj(raw_text)
        data = json.loads(json_str)
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
