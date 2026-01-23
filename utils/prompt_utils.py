# utils/prompt_utils.py

import json

from config.config import DatasetEnum

STAGE1_SYSTEM = """You are a rater who judges whether a text is appropriate or inappropriate.

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
    dataset_perspective: str,
    similar_texts: list[dict],
) -> str:

    sim_block_lines = []

    for i, ex in enumerate(similar_texts, start=1):
        ex_text = ex["text"]
        ex_label = ex["label"]
        ex_label_str = label_to_str(ex_label)

        sim_block_lines.append(f"{i}. {ex_text}, Label: {ex_label_str}")

    sim_block = "\n".join(sim_block_lines)

    return f"""{STAGE1_SYSTEM}
Perspective:
{dataset_perspective}

TEXT:
{text}

SIMILAR TEXTS (reference only):
{sim_block}
"""


def build_dataset_perspective(ds: DatasetEnum) -> str:
    if ds == DatasetEnum.HSOL:
        return """Perspective:
Texts are judged under three internal notions: group-targeted hate, offensive/insulting language that is not necessarily group-hate, and benign/neutral.
For binary use here, treat both hate and offensive/insulting as inappropriate, and benign/neutral as appropriate.
Do not decide purely by keywords; prioritize whether the text actually expresses hostility/derogation/insult toward a person or group.
"""

    if ds == DatasetEnum.HSDCD:
        return """Perspective:
Texts are social-media style English sentences (may include emojis/emoticons/hashtags/slang/contractions) labeled as hateful vs non-hateful.
Hateful: hateful/hostile/derogatory content toward people or groups; Non-hateful: benign/neutral content without hate.
Do not rely on keywords alone; judge intent and meaning in context (including emojis/slang usage).
For binary use: hateful=inappropriate (1), non-hateful=appropriate (0).
"""

    if ds == DatasetEnum.HateXplain:
        return """Perspective:
Texts follow three internal notions: hate, offensive, and normal.
Hate is hostility/derogation/dehumanization toward an identity community; offensive is abusive/insulting/harassing language that may target individuals or groups without being identity-hate; normal is benign/neutral.
For binary use here, treat hate+offensive as inappropriate and normal as appropriate.
Do not treat identity terms alone as evidence; focus on hostile/harassing intent and usage.
"""

    if ds == DatasetEnum.RealToxicityPrompts:
        return """Perspective:
Label toxicity using an automated toxicity scoring system that outputs a continuous score in [0, 1].
Treat score >= 0.5 as inappropriate (toxic) and score < 0.5 as appropriate (non-toxic).
Interpret the score as reflecting likelihood of rude/abusive/harassing or otherwise toxic language in the text.
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

    raise ValueError(f"Unsupported dataset for perspective: {ds}")


##############3



STAGE2_SYSTEM = """You are an analyst who reflects on a previous labeling decision.

Task:
- You are given:
  - the original TEXT,
  - the TRUE_LABEL (ground-truth),
  - the STAGE1_PRED_LABEL (previous prediction),
  - and the STAGE1_RATIONALE (why the previous prediction was made).
- Do NOT predict a new label.
- Analyze:
  - what parts of the TEXT and STAGE1_RATIONALE likely caused the previous prediction,
  - why that prediction is correct or incorrect compared to the TRUE_LABEL,
  - which cues in the TEXT are most useful to get the TRUE_LABEL right in the future.

Output rules:
- Write in English, 1–3 sentences.
- Focus on concrete textual cues (words, phrases, targets, tone, context).
- Output ONLY a single JSON object (no markdown, no code fences, no extra text).
- The JSON must be valid and complete in one line.
- Use exactly one key: evidence
- evidence: a short English paragraph describing your analysis.
- IMPORTANT: evidence must NOT contain any quote characters (no " and no ').
"""


def _format_rules(rules: list[str] | None) -> str:
    rules = [str(r).strip() for r in (rules or []) if str(r).strip()]
    if not rules:
        return "(no intervention rules provided)"
    return "\n".join([f"- {r}" for r in rules])


def build_rem_stage2_prompt(
    text: str,
    true_label: int,
    stage1_pred_label: int,
    stage1_rationale: str,
) -> str:
    true_label_str = label_to_str(true_label)
    stage1_pred_str = label_to_str(stage1_pred_label)

    return f"""{STAGE2_SYSTEM}

TEXT:
{text}

TRUE_LABEL:
{true_label_str}

STAGE1_PRED_LABEL:
{stage1_pred_str}

STAGE1_RATIONALE:
{stage1_rationale}
"""








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



GPT_INFER_SYSTEM = """You are a rater who judges whether a text is appropriate or inappropriate.

Instructions:
- Read the TEXT.
- Read the SIMILAR TEXTS as reference examples only.
- Use them as guidance, but always output the final label for the TARGET_TEXT only.
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

