# utils/gpt_client.py

from pydantic import BaseModel, Field

from openai import OpenAI


class RemStage1Out(BaseModel):
    pred_label: int = Field(..., description="0=normal, 1=inappropriate")
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    tags: list[str] = Field(default_factory=list)


class GPTClient:
    def __init__(self, model: str, temperature: float, max_output_tokens: int):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI()

    def call_api(self, prompt: str):
        response = self.client.responses.parse(
            model=self.model,
            input=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            text_format=RemStage1Out,
        )

        out = response.output_parsed

        return out.model_dump()
