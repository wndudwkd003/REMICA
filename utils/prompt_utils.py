# utils/prompt_utils.py

import json

from config.config import DatasetEnum

_STAGE1_SYSTEM = """You are a rater who judges whether a text is appropriate or inappropriate.

Instructions:
- Read the TEXT.
- Read the SIMILAR TEXTS as reference examples only.
- Decide one label: appropriate or inappropriate.
- Provide a short reason for your decision.

Critical output rules:
- Output ONLY a single JSON object (no markdown, no code fences, no extra text).
- The JSON must be valid and complete in one line.
- Use exactly two keys: pred_label, rationale
- pred_label: 0 or 1 (0=appropriate, 1=inappropriate)
- rationale: one short sentence
- IMPORTANT: rationale must NOT contain any quote characters (no " and no ').
"""


def label_to_str(label: int):
    return "appropriate" if label == 0 else "inappropriate"

def build_rem_stage1_prompt(
    text: str,
    similar_texts: list[dict],
) -> str:

    sim_block_lines = []

    for i, ex in enumerate(similar_texts, start=1):
        ex_text = ex["text"]
        ex_label = ex["label"]
        ex_label_str = label_to_str(ex_label)

        sim_block_lines.append(f"{i}. {ex_text}, Label: {ex_label_str}")

    sim_block = "\n".join(sim_block_lines)

    return f"""{_STAGE1_SYSTEM}

TEXT:
{text}

SIMILAR TEXTS (reference only):
{sim_block}
"""


def build_dataset_perspective(ds: DatasetEnum) -> str:
    """
    Stage2에서만 사용: 데이터셋별 라벨링 기준(관점) 텍스트.
    - 특정 논문/저자/연도/데이터셋 이름을 프롬프트에 노출하지 않음.
    - 여기서는 "관점/기준"만 제공 (final_label 같은 출력 스펙 언급 금지)
    """

    if ds == DatasetEnum.HSOL:
        return """Perspective:
Texts are judged under three internal notions: group-targeted hate, offensive/insulting language that is not necessarily group-hate, and benign/neutral.
For binary use here, treat both hate and offensive/insulting as inappropriate, and benign/neutral as appropriate.
Do not decide purely by keywords; prioritize whether the text actually expresses hostility/derogation/insult toward a person or group.
"""

    if ds == DatasetEnum.HateXplain:
        return """Perspective:
Texts follow three internal notions: hate, offensive, and normal.
Hate is hostility/derogation/dehumanization toward an identity community; offensive is abusive/insulting/harassing language that may target individuals or groups without being identity-hate; normal is benign/neutral.
For binary use here, treat hate+offensive as inappropriate and normal as appropriate.
Do not treat identity terms alone as evidence; focus on hostile/harassing intent and usage.
"""

    if ds == DatasetEnum.DiaSafety:
        return """Perspective:
Interpret the text as a bot reply in a dialogue setting; some replies are unsafe only when interpreted as a response.
Unsafe if it insults/mocks/escalates, encourages or normalizes serious risk (especially self-harm), gives definitive high-stakes medical/legal instructions, agrees with or amplifies hateful/violent/abusive context, or reinforces biased stereotypes.
Otherwise safe (de-escalating, cautious, refusing harmful requests, suggesting proper help when needed).
For binary use here, unsafe=inappropriate and safe=appropriate.
"""

    if ds == DatasetEnum.ToxiSpanSE:
        return """Perspective:
Texts are software-engineering code review comments. Domain terms like “kill”, “dead”, “crash”, “dump”, “bug” can be technical, not abusive.
Inappropriate if the comment shows antisocial/toxic behavior: insults/name-calling/personal attacks, threats, harassment, sexually explicit remarks, or profanities directed at someone.
Do not flag purely technical criticism or domain terms unless used as an insult/attack.
"""

    if ds == DatasetEnum.HSD:
        return """Perspective:
Judge whether the sentence is hate speech at sentence level.
Hate requires: a deliberate attack, directed at a group of people, motivated by the group’s identity.
Do not mark hate if it attacks only an individual, reports information, or discusses identity topics without a clear group-directed identity attack.
Offensive words alone are insufficient; verify the sentence forms a group-directed identity attack.
If ambiguous in isolation, treat as not-hate for this setting.
For binary use here, hate=inappropriate and not-hate=appropriate.
"""

    return """Perspective:
(TO BE FILLED)
"""


_STAGE2_OUTPUT_RULES = """Output ONLY one valid JSON object in ONE line (no markdown, no code fences).
Use exactly these keys:
- evidence: string
- memory: string

Constraints:
- evidence:
  - If CORRECT (stage1_pred == true_label): write the key evidence in TEXT that supports the true label (supporting evidence).
  - If WRONG (stage1_pred != true_label): write in this format:
    ERROR: <what likely misled Stage1> | MISSING: <what should have been checked to match the true label>
- memory: one-line reusable rule to apply next time (avoid quotes).
IMPORTANT: memory must NOT contain quote characters (no " and no ').
"""


def _format_rules(rules: list[str] | None) -> str:
    rules = [str(r).strip() for r in (rules or []) if str(r).strip()]
    if not rules:
        return "(no intervention rules provided)"
    return "\n".join([f"- {r}" for r in rules])


def build_rem_stage2_prompt(
    *,
    text: str,
    run_tag: str,  # "run0" | "run1" | "run2"
    stage1_pred_label: int | None = None,
    stage1_rationale: str | None = None,
    ica_rules: list[str] | None = None,  # run1/run2에서 제공되는 규칙(또는 변형된 규칙)
    true_label: int | None = None,  # 반성/강화용 정답(판정 shortcut 금지)
) -> str:
    """
    REM Stage 2 = Reflective Memory 생성 단계 (Perspective 없이).
    - pred_label은 여전히 출력하지만,
      true_label이 있으면 '판정 바꾸기'가 아니라 '근거/메모리 교정'에만 쓰도록 강제한다.
    """

    rules_block = _format_rules(ica_rules)

    schema = {
        "pred_label": 0,  # 0=appropriate, 1=inappropriate
        "evidence": "",  # 근거(맞/틀 분석 포함)
        "memory": "",  # 다음에 재사용 가능한 reflective memory (1~2문장)
        "used_rules": [],  # 실제 사용한 규칙 subset (run0이면 반드시 [])
    }

    gold_part = ""
    if true_label is not None:
        gold_part = f"""
Gold label (for reflection only):
- true_label: {int(true_label)}

Important:
- Do NOT use the gold label as a shortcut to change the classification.
- Use it ONLY to correct/strengthen reasoning: explain why the TEXT supports the true label.
""".strip()

    if run_tag == "run0":
        run_rule_constraint = """
Rules:
- No intervention rules are provided in this run.
- used_rules MUST be an empty list [].
""".strip()
    else:
        run_rule_constraint = """
Rules:
- Intervention rules are provided below.
- used_rules MUST be a subset of the provided rules (or empty if none apply).
""".strip()

    return f"""
You are generating Reflective Memory for safety/toxicity classification.

What you must do:
1) Predict pred_label for the TEXT (0=appropriate, 1=inappropriate).
2) Write EVIDENCE grounded in the TEXT (and rules if provided).
3) Write MEMORY: a compact reusable rule/checklist (1–2 sentences).
4) Fill used_rules with only the rules you actually used (subset). If no rules provided, used_rules must be [].

Definition:
- EVIDENCE must cite concrete cues from the TEXT (intent, target, violence, slurs, harassment, etc.).
- MEMORY must be generalizable: do not copy the text; write a reusable heuristic/check.

Strict constraints:
- No guessing beyond the text.
- Output ONLY one valid JSON object in one line.
- Use exactly the keys in the schema. Do not add any other keys.
- memory must NOT contain quote characters (no " and no ').

TEXT:
{text}

Stage1 hint (may be wrong/noisy):
- stage1_pred_label: {stage1_pred_label}
- stage1_rationale: {stage1_rationale}

{gold_part}

{run_rule_constraint}

Intervention rules:
{rules_block}

EVIDENCE writing rule:
- If true_label is provided and your pred_label matches it: write the key cues supporting the true label.
- If true_label is provided and your pred_label conflicts with it: write evidence in this format:
  ERROR: <what likely misled you or stage1> | MISSING: <what should be checked to match true_label>

Return ONLY this JSON schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()


def build_ica_prompt(turns, source_dataset: str, split: str, sid: str) -> str:
    """
    ICA prompt builder (Aligned with GPTClientICA.ICAOut)
    - Output JSON keys MUST be: context_summary, triggers, targets, rules
    - triggers/targets/rules MUST be JSON arrays of strings (NOT JSON-encoded strings)
    """

    def speaker_name(s):
        if s == 1 or str(s) == "1":
            return "A"
        if s == 2 or str(s) == "2":
            return "B"
        s = str(s).strip().upper()
        return "A" if s == "A" else "B"

    dialog = []
    for t in turns:
        sp = speaker_name(t.get("speaker"))
        txt = str(t.get("text", "")).strip()
        dialog.append(f"{sp}: {txt}")

    dialog_block = "\n".join(dialog)

    schema = {
        "context_summary": "string",
        "triggers": "array of strings",
        "targets": "array of strings",
        "rules": "array of strings",
    }

    return f"""
You are an extractor for safety/toxicity-aware dialogue classification.

What is an "intervention rule"?
- A compact, reusable decision heuristic distilled from the dialogue context.
- It states what contextual signals matter (triggers/targets) and how they should influence a later safety judgment.
- Example: "If the speaker expresses intent to harm a person/group, label as inappropriate."

Input:
- A 4-turn dialogue window with alternating speakers A/B.

Goal (use ONLY what is present in the dialogue):
1) Triggers: phrases/acts/intent signals that can affect a harmful/inappropriate judgment.
2) Targets: who/what is targeted (group/person/institution/animal/social group). If none, use an empty list.
3) Rules: 1–3 short, generalizable intervention rules derived from triggers/targets.
   Prefer if-then style (condition → decision hint). Keep them reusable.

Constraints:
- No guessing. Do not invent facts not supported by the dialogue.
- You MUST return JSON with exactly the keys below.
- triggers / targets / rules MUST each be a JSON array of strings (e.g., ["a","b"]), not a JSON-encoded string.

Meta:
- source_dataset: {source_dataset}
- split: {split}
- sid: {sid}

[4-turn dialogue]
{dialog_block}

Return ONLY a JSON object that matches this schema:
{json.dumps(schema, ensure_ascii=False)}
""".strip()
