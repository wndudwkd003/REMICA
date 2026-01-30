# utils/gpt_client.py

from __future__ import annotations

import json
import time
from typing import Any, Dict, Generic, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------
# 공용 헬퍼
# ---------------------------------------------------------


def extract_json_obj(raw: str) -> str:
    """
    모델이 출력한 문자열에서 첫 번째 JSON 객체 부분만 깔끔하게 잘라낸다.
    - 앞뒤 코드펜스( ```json ... ``` ) 제거
    - 가장 바깥 { ... } 구간만 추출
    """
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


def get_output_text(resp: Any) -> str:
    """
    Responses API 응답 객체(resp)에서 사람이 읽을 수 있는 텍스트만 추출.
    - responses.create(...) 호출 시 fallback용으로 사용.
    """
    # 일부 클라이언트 버전에서 직접 text를 주는 경우
    t = getattr(resp, "output_text", None)
    if isinstance(t, str) and t.strip():
        return t.strip()

    out = getattr(resp, "output", None)
    if out is None and isinstance(resp, dict):
        out = resp.get("output", None)
    out = out or []

    parts: list[str] = []
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


def summarize_output_types(resp: Any) -> str:
    """
    디버깅용: responses.* 응답 객체 안에 어떤 type들이 들어있는지 요약.
    에러 메시지에 참조용으로 붙여서 디버깅에 사용.
    """
    out = getattr(resp, "output", None)
    if out is None and isinstance(resp, dict):
        out = resp.get("output", None)
    out = out or []

    item_types: list[str] = []
    content_types: list[str] = []
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


def _is_unsupported_param_error(e: Exception, keys: list[str]) -> bool:
    msg = (str(e) or "").lower()
    if not msg:
        return False

    needles = [
        "unknown parameter",
        "unrecognized request argument",
        "not supported",
        "unsupported",
        "extra fields not permitted",
        "invalid request",
    ]
    if not any(n in msg for n in needles):
        return False

    return any(k.lower() in msg for k in keys)


# ---------------------------------------------------------
# GPTClient 본체
# ---------------------------------------------------------
class GPTClient(Generic[T]):
    def __init__(
        self,
        model: str,
        max_output_tokens: int,
        schema: Type[T],
        max_retries: int = 3,
        retry_sleep: float = 0.5,
        temperature: float = 0.0,  # ✅ 시그니처 유지
        top_p: float = 1.0,  # ✅ 시그니처 유지
    ) -> None:
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.max_retries = int(max_retries)
        self.retry_sleep = float(retry_sleep)

        self.temperature = float(temperature)
        self.top_p = float(top_p)

        self.client = OpenAI()
        self.schema = schema

        # ✅ 모델별로 "샘플링 파라미터 미지원"을 기억(한 번 실패하면 다음부터 omit)
        self._sampling_params_supported: bool | None = (
            None  # None=미확정, True/False=확정
        )

    def _make_sampling_kwargs(self) -> Dict[str, Any]:
        """
        - 지원한다고 확정(True)되면 포함
        - 미지원(False) 확정되면 미포함
        - 미확정(None)이면 일단 포함(그리고 에러나면 False로 확정)
        """
        if self._sampling_params_supported is False:
            return {}

        return {"temperature": self.temperature, "top_p": self.top_p}

    def _mark_sampling_supported(self, supported: bool) -> None:
        self._sampling_params_supported = supported

    def call_api(self, prompt: str) -> Dict[str, Any]:
        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_with_structured_outputs(prompt)
            except Exception as e1:
                last_err = e1
                try:
                    return self._call_with_raw_fallback(prompt)
                except Exception as e2:
                    last_err = e2

            if attempt < self.max_retries:
                time.sleep(self.retry_sleep)

        assert last_err is not None
        raise last_err

    def _call_with_structured_outputs(self, prompt: str) -> Dict[str, Any]:
        base_kwargs: Dict[str, Any] = dict(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=self.max_output_tokens,
            text_format=self.schema,
            **self._make_sampling_kwargs(),
        )

        try:
            resp = self.client.responses.parse(**base_kwargs)
            # ✅ 여기까지 왔으면(샘플링 kwargs 포함/미포함 모두) “지원됨”으로 간주해도 됨.
            # 단, 미확정 상태에서 포함했다가 성공한 경우만 True로 확정
            if "temperature" in base_kwargs or "top_p" in base_kwargs:
                self._mark_sampling_supported(True)

        except Exception as e:
            keys = ["temperature", "top_p"]
            # ✅ 미확정/지원(True) 상태에서 샘플링 파라미터가 원인으로 보이면 제거 재시도
            if _is_unsupported_param_error(e, keys) and any(
                k in base_kwargs for k in keys
            ):
                for k in keys:
                    base_kwargs.pop(k, None)

                # 이 모델은 샘플링 파라미터를 거부 → 캐시
                self._mark_sampling_supported(False)

                resp = self.client.responses.parse(**base_kwargs)
            else:
                raise

        out = getattr(resp, "output_parsed", None)
        if out is None:
            raise RuntimeError("responses.parse output_parsed=None")

        if isinstance(out, self.schema):
            return out.model_dump()

        if isinstance(out, list) and out and isinstance(out[0], self.schema):
            return out[0].model_dump()

        raise TypeError(f"Unexpected output_parsed type: {type(out)!r}")

    def _call_with_raw_fallback(self, prompt: str) -> Dict[str, Any]:
        base_kwargs: Dict[str, Any] = dict(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=self.max_output_tokens,
            **self._make_sampling_kwargs(),
        )

        try:
            resp = self.client.responses.create(**base_kwargs)
            if "temperature" in base_kwargs or "top_p" in base_kwargs:
                self._mark_sampling_supported(True)

        except Exception as e:
            keys = ["temperature", "top_p"]
            if _is_unsupported_param_error(e, keys) and any(
                k in base_kwargs for k in keys
            ):
                for k in keys:
                    base_kwargs.pop(k, None)

                self._mark_sampling_supported(False)

                resp = self.client.responses.create(**base_kwargs)
            else:
                raise

        raw_text = get_output_text(resp)  # 기존 함수 사용
        if not raw_text:
            raise ValueError("empty_output_text")

        json_str = extract_json_obj(raw_text)  # 기존 함수 사용
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"json_decode_error: {e} | raw_fragment={json_str[:200]}")

        try:
            checked = self.schema.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"schema_validation_error: {e}")

        return checked.model_dump()
