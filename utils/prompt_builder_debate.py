# utils/prompt_builder_debate.py

import json
from enums.out_schema import GPTInferOut, AgentAOut, AgentBOut
from enums.structured_enum import (
    FACTOR_TOKENS,
    TARGET_TYPE_TOKENS,
    TARGET_GROUP_TOKENS,
    STANCE_TOKENS,
)
from utils.prompt_builder_dataset import build_dataset_perspective
from config.config import ChainEnum


def header(title: str) -> str:
    return f"[{title}]"


def bullets(items, prefix: str = "-") -> str:
    out = []
    for x in items or []:
        x = (x or "").strip()
        if x:
            out.append(f"{prefix} {x}")
    return "\n".join(out).strip()


def section(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"{header(title)}\n{body}\n"


def join_sections(parts) -> str:
    out = []
    for p in parts or []:
        p = (p or "").strip()
        if p:
            out.append(p)
    return "\n\n".join(out).strip()


def pretty_json(x) -> str:
    return json.dumps(x, ensure_ascii=False, indent=2)


def safe_dump(x):
    if hasattr(x, "model_dump"):
        return x.model_dump()
    return x


def format_schema(schema_obj) -> str:
    if isinstance(schema_obj, (dict, list)):
        return pretty_json(schema_obj)
    return str(schema_obj)


# =========================================================
# Instruct / Task / Input / Constraints
# =========================================================


def make_instruct_lines(role_text, perspective_text, output_schema_text, extra=None):
    lines = []
    if role_text:
        lines.append(f"Role: {role_text.strip()}")

    lines.append(
        "Label convention: pred_label is an integer. 0=appropriate, 1=inappropriate"
    )

    if perspective_text:
        lines.append("Dataset perspective:")
        lines.append(perspective_text.strip())

    if output_schema_text:
        lines.append("Output format: return a single JSON object matching this schema:")
        lines.append(format_schema(output_schema_text).strip())

    if extra:
        if isinstance(extra, str):
            extra = [extra]
        for x in extra:
            x = (x or "").strip()
            if x:
                lines.append(x)

    return lines


def make_instruct(role_text, perspective_text, output_schema_text, extra=None):
    return bullets(
        make_instruct_lines(role_text, perspective_text, output_schema_text, extra)
    )


def make_task(items):
    return bullets(items)


def make_input(items):
    return bullets(items)


def make_common_constraints():
    return [
        "pred_label MUST be exactly 0 (appropriate) or 1 (inappropriate).",
        "All free-text fields MUST NOT contain any quote characters (no \" and no ').",
        "Do NOT output any extra text before or after the JSON object.",
    ]


def make_constraints(extra=None, remove=None):
    base = make_common_constraints()

    if remove:
        if isinstance(remove, str):
            remove = [remove]
        remove_set = set(remove)
        base = [x for x in base if x not in remove_set]

    if extra:
        if isinstance(extra, str):
            extra = [extra]
        for x in extra:
            x = (x or "").strip()
            if x and x not in base:
                base.append(x)

    return bullets(base)


def assemble_prompt(
    role_text,
    perspective_text,
    task_items,
    input_items,
    output_schema_text,
    constraints_extra=None,
    constraints_remove=None,
    instruct_extra=None,
):
    instruct = make_instruct(
        role_text, perspective_text, output_schema_text, extra=instruct_extra
    )
    task = make_task(task_items)
    inp = make_input(input_items)
    cons = make_constraints(extra=constraints_extra, remove=constraints_remove)

    parts = [
        section("Instruct", instruct),
        section("Task", task),
        section("Input", inp),
        section("Constraints", cons),
    ]
    return join_sections(parts)


# =========================================================
# Similar examples
# =========================================================


def format_example_label(label_value):
    try:
        v = int(label_value)
        if v == 0:
            return "0 (appropriate)"
        if v == 1:
            return "1 (inappropriate)"
        return str(v)
    except:
        return label_value


def get_out(ex, key):
    step = ex.get(key)
    if isinstance(step, dict):
        return step.get("output")
    return None


def format_similar_examples(similar_examples, chain_mode: ChainEnum) -> str:
    lines = []
    for i, ex in enumerate(similar_examples, start=1):
        text = ex["text"]
        label_str = format_example_label(ex["label"])

        lines.append(f"({i}). text: {text}, label: {label_str}")

        if chain_mode == ChainEnum.NONE:
            lines.append("")  # separator
            continue

        if chain_mode == ChainEnum.CHAIN_OF_DEBATE:
            order = ("A1", "B1", "A2", "B2")

        elif chain_mode == ChainEnum.CHAIN_OF_EXPERT:
            order = ("A1", "A2", "B1", "B2")

        for step_name in order:
            out_mem = get_out(ex, step_name)  # extract output dict
            if out_mem is not None:
                lines.append(f"{step_name} output:\n{pretty_json(out_mem)}")

        lines.append("")  # separator

    return "\n".join(lines).strip()


# =========================================================
# Persona
# =========================================================


def persona_agent_A():
    return (
        "You are Agent A: a guideline-centered classifier that decides whether a target text is "
        "appropriate (0) or inappropriate (1) for hate/toxicity detection. "
        "You must distill the dataset perspective into a compact guideline_context that helps reach the correct decision, "
        "then provide rationale grounded in that guideline_context."
    )


def persona_agent_B():
    return (
        "You are Agent B: a structure-centered classifier that decides whether a target text is "
        "appropriate (0) or inappropriate (1) for hate/toxicity detection. "
        "You must produce structured_context first, then derive pred_label from it and explain your rationale."
    )


# =========================================================
# Token blocks (allowed structured tokens)
# =========================================================


def token_block(title, xs):
    xs = xs or []
    body = "\n".join([f"- {x}" for x in xs]).strip()
    return f"{title}\n{body}".strip()


def actual_label_block(actual_label):
    """
    실제 라벨은 '정답'으로 제공되며, A2/B2에서 반성/강화(reflection) 용도로 사용.
    '정답 타당성'을 증명하지 말고, 정답을 참고하여 출력과 설명을 그에 맞추도록 지시.
    """
    return (
        "Actual label (ground truth):\n"
        f"{actual_label}\n\n"
        "Instruction:\n"
        "- The Actual label is the ground-truth answer.\n"
        "- Use it as a reference constraint for reflection/reinforcement.\n"
        "- Output pred_label consistent with the Actual label.\n"
        "- Do NOT justify why the Actual label is correct; instead, align your guideline_context/structured_context and rationale with it."
    )


# =========================================================
# Agent A
#  - prev_A_out / prev_B_out 의견을 반영/비판하여 rationale에 포함 가능
#  - actual_label_intervention=True면 (A2에서) 정답 라벨을 '참고 제약'으로 제공
# =========================================================


def build_prompt_agent_A(
    target_text,
    dataset,
    similar_examples: list,
    prev_A_out=None,
    prev_B_out=None,
    actual_label_intervention=False,
    actual_label=None,
    step_tag="A",
):
    schema_desc = AgentAOut.model_json_schema()
    perspective = build_dataset_perspective(dataset)

    examples_str = format_similar_examples(similar_examples, ChainEnum.NONE)

    task_items = [
        f"{step_tag}: Decide whether the target text is appropriate (0) or inappropriate (1).",
        "Write guideline_context: a compact distilled guideline derived from the dataset perspective that is sufficient to make the decision.",
        "Write rationale grounded in guideline_context and dataset perspective. If previous outputs are provided, reflect them in rationale (support or rebut).",
    ]

    input_items = [
        "Similar examples:\n" + str(examples_str),
        "Target text:\n" + str(target_text),
    ]

    if examples_str.strip() == "":
        input_items.pop(0)  # remove 'examples_str' if empty

    if prev_A_out is not None:
        input_items.append(
            "Previous Agent A output:\n" + pretty_json(safe_dump(prev_A_out))
        )

    if prev_B_out is not None:
        input_items.append(
            "Previous Agent B output:\n" + pretty_json(safe_dump(prev_B_out))
        )

    if actual_label_intervention:
        input_items.append(actual_label_block(actual_label))

    constraints_extra = [
        "guideline_context MUST be present and must be short (compact distilled guideline).",
        "rationale MUST be present and grounded in guideline_context and dataset perspective.",
    ]

    if actual_label_intervention:
        constraints_extra.append("pred_label MUST match the Actual label exactly.")
        constraints_extra.append(
            "guideline_context and rationale MUST be consistent with the final pred_label."
        )

    return assemble_prompt(
        role_text=persona_agent_A(),
        perspective_text=perspective,
        task_items=task_items,
        input_items=input_items,
        output_schema_text=schema_desc,
        constraints_extra=constraints_extra,
    )


# =========================================================
# Agent B few-shot (StructuredContext)
# =========================================================


def build_few_shot_examples_agent_B_text():
    ex1 = """
[Example 1 – Inappropriate (Harassment)]
Target text:
you are a useless idiot

Expected output:
{
  "pred_label": 1,
  "structured_context": {
    "factor_tokens": ["HARASSMENT_INSULT"],
    "target_type": "INDIVIDUAL",
    "target_group": "NONE",
    "stance": "ENDORSE",
    "evidence_spans": ["useless idiot", "you"]
  },
  "rationale": "This is a direct personal insult toward an individual, which is harassment and inappropriate."
}
""".strip()

    ex2 = """
[Example 2 – Appropriate (Neutral)]
Target text:
cats love eating candles

Expected output:
{
  "pred_label": 0,
  "structured_context": {
    "factor_tokens": ["NEUTRAL_DAILY"],
    "target_type": "GROUP",
    "target_group": "OTHER_GROUP",
    "stance": "NEUTRAL",
    "evidence_spans": ["cats"]
  },
  "rationale": "Odd but benign statement with no hate, harassment, sexual content, or threats; appropriate."
}
""".strip()

    return "\n\n".join([ex1, ex2]).strip()


# =========================================================
# Agent B
#  - prev_A_out / prev_B_out 의견 반영 가능
#  - actual_label_intervention=True면 (B2에서) 정답 라벨을 '참고 제약'으로 제공
# =========================================================


def build_prompt_agent_B(
    target_text,
    dataset,
    similar_examples,
    prev_A_out=None,
    prev_B_out=None,
    actual_label_intervention=False,
    actual_label=None,
    step_tag="B",
):
    schema_desc = AgentBOut.model_json_schema()
    perspective = build_dataset_perspective(dataset)

    few_shot_str = build_few_shot_examples_agent_B_text()
    examples_str = format_similar_examples(similar_examples, ChainEnum.NONE)

    task_items = [
        f"{step_tag}: Build structured_context and decide pred_label.",
        "structured_context MUST be consistent with the target text and chosen only from allowed token lists.",
        "Write rationale grounded in structured_context and dataset perspective. If previous outputs are provided, reflect them in rationale (support or rebut).",
    ]

    input_items = [
        "Few-shot examples:\n" + few_shot_str,
        "Similar examples:\n" + str(examples_str),
        "Target text:\n" + str(target_text),
    ]

    if prev_A_out is not None:
        input_items.append(
            "Previous Agent A output:\n" + pretty_json(safe_dump(prev_A_out))
        )

    if prev_B_out is not None:
        input_items.append(
            "Previous Agent B output:\n" + pretty_json(safe_dump(prev_B_out))
        )

    if actual_label_intervention:
        input_items.append(actual_label_block(actual_label))

    instruct_extra = [
        token_block("Allowed FACTOR_TOKENS:", FACTOR_TOKENS),
        token_block("Allowed TARGET_TYPE_TOKENS:", TARGET_TYPE_TOKENS),
        token_block("Allowed TARGET_GROUP_TOKENS:", TARGET_GROUP_TOKENS),
        token_block("Allowed STANCE_TOKENS:", STANCE_TOKENS),
    ]

    constraints_extra = [
        "structured_context MUST be present.",
        "structured_context.factor_tokens MUST be chosen only from FACTOR_TOKENS (list; may contain multiple).",
        "structured_context.target_type MUST be chosen only from TARGET_TYPE_TOKENS (exactly one).",
        "structured_context.target_group MUST be chosen only from TARGET_GROUP_TOKENS (exactly one).",
        "If structured_context.target_type != GROUP, structured_context.target_group MUST be NONE.",
        "structured_context.stance MUST be chosen only from STANCE_TOKENS (exactly one).",
        "structured_context.evidence_spans MUST be a list of contiguous text spans copied from the target text.",
        "rationale MUST be present and grounded in structured_context and dataset perspective.",
    ]

    if actual_label_intervention:
        constraints_extra.append("pred_label MUST match the Actual label exactly.")
        constraints_extra.append(
            "structured_context MUST be consistent with the final pred_label and the target text."
        )

    return assemble_prompt(
        role_text=persona_agent_B(),
        perspective_text=perspective,
        task_items=task_items,
        input_items=input_items,
        output_schema_text=schema_desc,
        constraints_extra=constraints_extra,
        instruct_extra=instruct_extra,
    )


# =========================================================
# 테스트(평가)용: A/B 출력 기반 최종 판정
# =========================================================


def build_prompt_test_infer(
    target_text,
    dataset,
    chain_mode: ChainEnum,
    memories: list[dict],
):
    """
    통합 최종 추론 프롬프트.

    memories item 예시:
      {
        "text": str,
        "label": int,
        "A1": {"prompt": str, "output": dict} | None,
        "A2": {"prompt": str, "output": dict} | None,
        "B1": {"prompt": str, "output": dict} | None,
        "B2": {"prompt": str, "output": dict} | None,
      }

    chain_mode:
      - NONE: memory의 텍스트/라벨만 제시 (A/B 출력은 무시)
      - CHAIN_OF_DEBATE: A1 -> B1 -> A2 -> B2
      - CHAIN_OF_EXPERT: A1 -> A2 -> B1 -> B2
    """

    schema_desc = GPTInferOut.model_json_schema()
    perspective = build_dataset_perspective(dataset)

    role = (
        "You are a binary classifier that decides whether a target text is appropriate (0) or "
        "inappropriate (1) for hate/toxicity detection."
    )

    task_items = [
        "Decide the final pred_label using the dataset perspective.",
        "Provide a short rationale grounded in the dataset perspective.",
    ]

    # -------------------------
    # format memories
    # -------------------------

    memories_str = format_similar_examples(memories, chain_mode)

    input_items = [
        "Target text:\n" + str(target_text),
        "Memories:\n" + memories_str,
    ]

    return assemble_prompt(
        role_text=role,
        perspective_text=perspective,
        task_items=task_items,
        input_items=input_items,
        output_schema_text=schema_desc,
        constraints_extra=None,
        constraints_remove=None,
        instruct_extra=None,
    )


def build_prompt_chain_step(
    step_index: int,
    target_text,
    dataset,
    similar_examples: list,
    out_A1=None,
    out_B1=None,
    out_A2=None,
    out_B2=None,
    actual_label_intervention: bool = False,
    actual_label=None,
    chain_mode: ChainEnum = ChainEnum.CHAIN_OF_DEBATE,
):

    # -------------------------
    # CHAIN_OF_EXPERT
    # -------------------------
    if chain_mode == ChainEnum.CHAIN_OF_EXPERT:
        if step_index == 1:
            return "A1", build_prompt_agent_A(
                target_text=target_text,
                dataset=dataset,
                similar_examples=similar_examples,
                prev_A_out=None,
                prev_B_out=None,
                actual_label_intervention=False,
                actual_label=None,
                step_tag="A1",
            )

        if step_index == 2:
            return "A2", build_prompt_agent_A(
                target_text=target_text,
                dataset=dataset,
                similar_examples=similar_examples,
                prev_A_out=out_A1,
                prev_B_out=None,
                actual_label_intervention=actual_label_intervention,
                actual_label=actual_label,
                step_tag="A2",
            )

        if step_index == 3:
            return "B1", build_prompt_agent_B(
                target_text=target_text,
                dataset=dataset,
                similar_examples=similar_examples,
                prev_A_out=None,
                prev_B_out=None,
                actual_label_intervention=False,
                actual_label=None,
                step_tag="B1",
            )

        if step_index == 4:
            return "B2", build_prompt_agent_B(
                target_text=target_text,
                dataset=dataset,
                similar_examples=similar_examples,
                prev_A_out=None,
                prev_B_out=out_B1,
                actual_label_intervention=actual_label_intervention,
                actual_label=actual_label,
                step_tag="B2",
            )

        raise ValueError(f"Unsupported step_index for CHAIN_OF_EXPERT: {step_index}")

    # -------------------------
    # CHAIN_OF_DEBATE
    # -------------------------
    if chain_mode == ChainEnum.CHAIN_OF_DEBATE:
        if step_index == 1:
            return "A1", build_prompt_agent_A(
                target_text=target_text,
                dataset=dataset,
                similar_examples=similar_examples,
                prev_A_out=None,
                prev_B_out=None,
                actual_label_intervention=False,
                actual_label=None,
                step_tag="A1",
            )

        if step_index == 2:
            return "B1", build_prompt_agent_B(
                target_text=target_text,
                dataset=dataset,
                similar_examples=similar_examples,
                prev_A_out=out_A1,
                prev_B_out=None,
                actual_label_intervention=False,
                actual_label=None,
                step_tag="B1",
            )

        if step_index == 3:
            return "A2", build_prompt_agent_A(
                target_text=target_text,
                dataset=dataset,
                similar_examples=similar_examples,
                prev_A_out=out_A1,
                prev_B_out=out_B1,
                actual_label_intervention=actual_label_intervention,
                actual_label=actual_label,
                step_tag="A2",
            )

        if step_index == 4:
            return "B2", build_prompt_agent_B(
                target_text=target_text,
                dataset=dataset,
                similar_examples=similar_examples,
                prev_A_out=out_A2,
                prev_B_out=out_B1,
                actual_label_intervention=actual_label_intervention,
                actual_label=actual_label,
                step_tag="B2",
            )

        raise ValueError(f"Unsupported step_index for CHAIN_OF_DEBATE: {step_index}")

    if chain_mode == ChainEnum.NONE:
        raise ValueError("ChainEnum.NONE does not support chain steps (A1/B1/A2/B2).")

    raise ValueError(f"Unsupported chain_mode: {chain_mode}")
