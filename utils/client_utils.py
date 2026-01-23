# utils/gpt_client.py
from __future__ import annotations

import json
import time
from typing import Any, Dict, Generic, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, Field

# ------------------------
# 예시 스키마들
# ------------------------

class RemStage1Out(BaseModel):
    pred_label: int = Field(..., description="0=appropriate, 1=inappropriate")
    rationale: str = Field(..., description="설명")


class RemStage2Out(BaseModel):
    evidence: str = Field(...)

class GPTInferOut(BaseModel):
    pred_label: int = Field(..., description="0 or 1")
    rationale: str = Field(
        ..., description="Optional short rationale; may be empty string"
    )



# ------------------------
# 공용 유틸
# ------------------------

def extract_json_obj(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty_output_text")

    # 코드펜스 제거
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    l = raw.find("{")
    r = raw.rfind("}")
    if l == -1 or r == -1 or r <= l:
        raise ValueError(f"no_json_object: {raw[:200]}")
    return raw[l : r + 1]


def get_output_text(resp) -> str:
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    out = getattr(resp, "output", None)
    if out is None and isinstance(resp, dict):
        out = resp.get("output", None)
    out = out or []

    parts = []
    for item in out:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content", None)
        content = content or []

        for c in content:
            c_type = getattr(c, "type", None)
            if c_type is None and isinstance(c, dict):
                c_type = c.get("type", None)

            if c_type in ("output_text", "text"):
                txt = getattr(c, "text", None)
                if txt is None and isinstance(c, dict):
                    txt = c.get("text", None)
                if isinstance(txt, str) and txt:
                    parts.append(txt)

    return "".join(parts).strip()


def summarize_output_types(resp) -> str:
    out = getattr(resp, "output", None)
    if out is None and isinstance(resp, dict):
        out = resp.get("output", None)
    out = out or []

    item_types = []
    content_types = []
    for item in out:
        it = getattr(item, "type", None)
        if it is None and isinstance(item, dict):
            it = item.get("type", None)
        if it:
            item_types.append(str(it))

        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content", None)
        content = content or []

        for c in content:
            ct = getattr(c, "type", None)
            if ct is None and isinstance(c, dict):
                ct = c.get("type", None)
            if ct:
                content_types.append(str(ct))

    return f"item_types={item_types[:8]} content_types={content_types[:12]}"


# ------------------------
# 공용 GPTClient (Stage1/2/ICA 다 여기로)
# ------------------------

T = TypeVar("T", bound=BaseModel)


class GPTClient(Generic[T]):
    def __init__(
        self,
        model: str,
        max_output_tokens: int,
        schema: Type[T],           # ← Stage1/2/ICA마다 다른 BaseModel
        max_retries: int = 3,
        retry_sleep: float = 0.5,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.max_retries = int(max_retries)
        self.retry_sleep = float(retry_sleep)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.client = OpenAI()
        self.schema = schema

    def call_api(self, prompt: str) -> Dict[str, Any]:
        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # 1) Structured outputs (parse) 먼저 시도
                resp = self.client.responses.parse(
                    model=self.model,
                    input=[{"role": "user", "content": prompt}],
                    max_output_tokens=self.max_output_tokens,
                    text_format=self.schema,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                out = resp.output_parsed
                if out is None:
                    raise RuntimeError(
                        f"responses.parse output_parsed=None | {summarize_output_types(resp)}"
                    )
                return out.model_dump()

            except Exception as e1:
                last_err = e1

                # 2) 실패하면 raw 생성 + 수동 JSON 파싱 fallback
                try:
                    resp = self.client.responses.create(
                        model=self.model,
                        input=[{"role": "user", "content": prompt}],
                        max_output_tokens=self.max_output_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                    raw_text = get_output_text(resp)
                    if not raw_text:
                        raise ValueError(
                            "empty_output_text | " + summarize_output_types(resp)
                        )

                    json_str = extract_json_obj(raw_text)
                    data = json.loads(json_str)
                    checked = self.schema.model_validate(data)
                    return checked.model_dump()

                except Exception as e2:
                    last_err = e2

            if attempt < self.max_retries:
                time.sleep(self.retry_sleep)

        # 모든 재시도 실패 시 마지막 에러 리레이즈
        raise last_err
