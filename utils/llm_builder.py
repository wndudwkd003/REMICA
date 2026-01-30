# utils/llm_builder.py

from __future__ import annotations

from typing import Type, TypeVar

from pydantic import BaseModel

from config.config import Config
from client.gpt_client import GPTClient
from client.claude_client import ClaudeClient

T = TypeVar("T", bound=BaseModel)


def build_agent_client(
    config: Config,
    schema: Type[T],
    role_key: str,
) -> GPTClient[T] | ClaudeClient[T]:

    model_name = config.api_model.value
    max_output_tokens = config.max_new_tokens.get(role_key, 2048)

    max_retries = config.max_retries
    temperature = config.temperature
    top_p = config.top_p

    # GPT family
    if model_name.startswith("gpt-"):
        return GPTClient[T](
            model=model_name,
            max_output_tokens=max_output_tokens,
            schema=schema,
            max_retries=max_retries,
            temperature=temperature,
            top_p=top_p,
        )

    # Claude family
    if model_name.startswith("claude-"):
        return ClaudeClient[T](
            model=model_name,
            max_output_tokens=max_output_tokens,
            schema=schema,
            max_retries=max_retries,
            temperature=temperature,
            top_p=top_p,
        )

    raise ValueError(f"Unknown model backend for model: {model_name}")
