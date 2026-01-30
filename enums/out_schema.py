# enums/out_schema.py

from pydantic import BaseModel, Field


class AgentAOut(BaseModel):
    pred_label: int = Field(..., description="0=appropriate, 1=inappropriate")
    guideline_context: str = Field(..., description="Compact self-made guideline")
    rationale: str = Field(
        ..., description="Guideline-based rationale (may reflect prev outputs)"
    )


class StructuredContext(BaseModel):
    factor_tokens: list[str] = Field(
        default_factory=list, description="Chosen from FACTOR_TOKENS"
    )
    target_type: str = Field(..., description="One of TARGET_TYPE_TOKENS")
    target_group: str = Field(..., description="One of TARGET_GROUP_TOKENS (or NONE)")
    stance: str = Field(..., description="One of STANCE_TOKENS")
    evidence_spans: list[str] = Field(
        default_factory=list, description="Contiguous spans copied from text"
    )


class AgentBOut(BaseModel):
    pred_label: int = Field(..., description="0=appropriate, 1=inappropriate")
    structured_context: StructuredContext = Field(
        default_factory=StructuredContext, description="Structured reading"
    )
    rationale: str = Field(
        ..., description="Structure-based rationale (may reflect prev outputs)"
    )


class GPTInferOut(BaseModel):
    pred_label: int = Field(..., description="0=appropriate, 1=inappropriate")
    rationale: str = Field(..., description="Short rationale")


class GPTDirectInferOut(BaseModel):
    pred_label: int = Field(..., description="0=appropriate, 1=inappropriate")
