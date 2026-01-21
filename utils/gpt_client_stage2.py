# utils/gpt_client_stage2.py
from __future__ import annotations

import json
import time

from openai import OpenAI
from pydantic import BaseModel, Field


class RemStage2Out(BaseModel):
    pred_label: int = Field(...)
    evidence: str = Field(...)
    memory: str = Field(...)
    run_tag: str = Field(...)
    used_rules: list[str] = Field(default_factory=list)


def extract_json_obj(raw: str) -> str:
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

    def call_api(self, prompt: str) -> dict:
        last_err = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.responses.parse(
                    model=self.model,
                    input=[{"role": "user", "content": prompt}],
                    max_output_tokens=self.max_output_tokens,
                    text_format=RemStage2Out,
                )
                out = resp.output_parsed
                if out is None:
                    raise RuntimeError(
                        f"responses.parse output_parsed=None | {summarize_output_types(resp)}"
                    )
                d = out.model_dump()
                return d

            except Exception as e1:
                last_err = e1

                try:
                    resp = self.client.responses.create(
                        model=self.model,
                        input=[{"role": "user", "content": prompt}],
                        max_output_tokens=self.max_output_tokens,
                    )
                    raw_text = get_output_text(resp)
                    if not raw_text:
                        raise ValueError(
                            "empty_output_text | " + summarize_output_types(resp)
                        )

                    json_str = extract_json_obj(raw_text)
                    data = json.loads(json_str)
                    checked = RemStage2Out.model_validate(data)
                    return checked.model_dump()

                except Exception as e2:
                    last_err = e2

            if attempt < self.max_retries:
                time.sleep(self.retry_sleep)

        raise last_err
