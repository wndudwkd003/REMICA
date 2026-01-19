# utils/gpt_client_ica.py
from __future__ import annotations

import json
import time

from openai import OpenAI
from pydantic import BaseModel, Field


class ICAOut(BaseModel):
    context_summary: str = Field(...)
    triggers: list[str] = Field(...)
    targets: list[str] = Field(...)
    rules: list[str] = Field(...)


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
        raise ValueError(f"no_json_object: {raw[:200]}")
    return raw[l : r + 1]


def _get_output_text(resp) -> str:
    # 1) SDK가 제공하는 합쳐진 output_text가 있으면 우선 사용
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    # 2) output[*].content[*] 순회하며 type별로 수집
    out = getattr(resp, "output", None)
    if isinstance(out, list):
        chunks = []
        refusals = []
        for item in out:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for c in content:
                ctype = getattr(c, "type", None)

                # 가장 표준
                if ctype in ("output_text", "text"):
                    txt = getattr(c, "text", None)
                    if isinstance(txt, str) and txt.strip():
                        chunks.append(txt.strip())
                        continue

                # 거절/필터
                if ctype == "refusal":
                    rf = getattr(c, "refusal", None)
                    if isinstance(rf, str) and rf.strip():
                        refusals.append(rf.strip())
                        continue

                # 일부 SDK에서는 type 없이 text만 있을 수도 있어서 fallback
                txt = getattr(c, "text", None)
                if isinstance(txt, str) and txt.strip():
                    chunks.append(txt.strip())

        if chunks:
            return "\n".join(chunks)

        # refusal만 있는 경우도 "빈 출력"로 치면 안 됨 → 이유 노출
        if refusals:
            return "[REFUSAL]\n" + "\n".join(refusals)

    return ""


class GPTClientICA:
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

    def _call_parse(self, prompt: str) -> dict:
        resp = self.client.responses.parse(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=self.max_output_tokens,
            text_format=ICAOut,
        )
        out = resp.output_parsed
        if out is None:
            raw = _get_output_text(resp)
            raise RuntimeError(f"parse_output_parsed_none | raw_head={raw[:300]}")
        return out.model_dump()

    def _call_raw_and_parse(self, prompt: str) -> dict:
        resp = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=self.max_output_tokens,
        )

        raw_text = _get_output_text(resp)
        json_str = _extract_json_obj(raw_text)

        try:
            data = json.loads(json_str)
        except Exception as e:
            raise RuntimeError(
                f"raw_json_decode_fail: {type(e).__name__}: {e} | raw_head={raw_text[:300]}"
            )

        checked = ICAOut.model_validate(data)
        return checked.model_dump()

    def call_api(self, prompt: str) -> dict:
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
