# utils/claude_client.py

import json
import time
from typing import Any, Dict, Generic, Type, TypeVar

from anthropic import Anthropic, BadRequestError
from pydantic import BaseModel, ValidationError

from client.gpt_client import extract_json_obj

T = TypeVar("T", bound=BaseModel)


def _is_sampling_param_unsupported(err: Exception) -> bool:
    """
    temperature/top_p 가 모델에서 허용되지 않아 발생한 BadRequestError인지 판별.
    Anthropic SDK의 공식 에러 타입을 활용.
    """
    if not isinstance(err, BadRequestError):
        return False

    # BadRequestError의 body 또는 message에서 파라미터 관련 에러 확인
    error_msg = ""

    # body 속성이 dict인 경우
    if hasattr(err, "body") and isinstance(err.body, dict):
        error_info = err.body.get("error", {})
        error_msg = error_info.get("message", "").lower()

    # message 속성 확인
    if not error_msg and hasattr(err, "message"):
        error_msg = str(err.message).lower()

    # 최종적으로 str(err) 확인
    if not error_msg:
        error_msg = str(err).lower()

    # temperature 또는 top_p 관련 에러인지 확인
    mentions_param = (
        ("temperature" in error_msg) or ("top_p" in error_msg) or ("topp" in error_msg)
    )
    looks_like_unsupported = (
        ("unsupported" in error_msg)
        or ("not supported" in error_msg)
        or ("not allowed" in error_msg)
        or ("not permitted" in error_msg)
        or ("unknown" in error_msg)
        or ("unrecognized" in error_msg)
        or ("invalid" in error_msg)
        or ("extra inputs" in error_msg)
        or ("cannot both be specified" in error_msg)
    )

    return mentions_param and looks_like_unsupported


class ClaudeClient(Generic[T]):
    """
    Anthropic Claude API용 공용 클라이언트.

    - 입력: prompt (str)
    - 출력: 지정한 Pydantic 스키마 T를 만족하는 dict
    - GPTClient와 동일한 인터페이스 제공
    """

    def __init__(
        self,
        model: str,
        max_output_tokens: int,
        schema: Type[T],
        max_retries: int = 3,
        retry_sleep: float = 0.5,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.model = model
        self.max_output_tokens = int(max_output_tokens)
        self.max_retries = int(max_retries)
        self.retry_sleep = float(retry_sleep)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.client = Anthropic()  # ANTHROPIC_API_KEY 환경변수 사용
        self.schema = schema

    def _call_messages_create(self, prompt: str, *, use_sampling_params: bool):
        """
        Claude messages.create 호출 래퍼.
        use_sampling_params=False 이면 temperature를 아예 넘기지 않습니다.
        """
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_output_tokens,
            "system": (
                "You are a JSON-only API. "
                "Respond with a single valid JSON object that matches the given schema. "
                "Do not include any explanation or extra text."
            ),
            "messages": [{"role": "user", "content": prompt}],
        }

        if use_sampling_params:
            # temperature만 사용 (Claude 4.5 모델은 temperature와 top_p 동시 사용 불가)
            kwargs["temperature"] = self.temperature

        return self.client.messages.create(**kwargs)

    def call_api(self, prompt: str) -> Dict[str, Any]:
        """
        prompt를 받아 Claude API를 호출하고,
        Pydantic 스키마로 검증된 dict를 반환.

        max_retries 회수 내에서 재시도.
        """
        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                # 1) 기본: sampling 파라미터 포함 호출
                try:
                    resp = self._call_messages_create(prompt, use_sampling_params=True)
                except BadRequestError as e:
                    # 2) temperature 미지원 모델이면 파라미터 제거 후 같은 attempt에서 즉시 1회 재호출
                    if _is_sampling_param_unsupported(e):
                        resp = self._call_messages_create(
                            prompt, use_sampling_params=False
                        )
                    else:
                        raise

                # 텍스트 추출
                parts: list[str] = []
                for block in resp.content:
                    if getattr(block, "type", None) == "text":
                        parts.append(block.text)

                raw_text = "".join(parts).strip()
                if not raw_text:
                    raise ValueError("empty_output_text_claude")

                # JSON 추출 및 파싱
                json_str = extract_json_obj(raw_text)
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"json_decode_error: {e} | raw_fragment={json_str[:200]}"
                    )

                # 스키마 검증
                try:
                    checked = self.schema.model_validate(data)
                except ValidationError as e:
                    raise ValueError(f"schema_validation_error: {e}")

                return checked.model_dump()

            except Exception as e:
                last_err = e

            # 다음 재시도
            if attempt < self.max_retries:
                time.sleep(self.retry_sleep)

        # 모든 재시도 실패
        assert last_err is not None
        raise last_err
