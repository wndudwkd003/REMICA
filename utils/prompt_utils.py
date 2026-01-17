# utils/prompt_utils.py

from typing import List
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
- Use exactly these keys: pred_label, confidence, rationale
- pred_label: 0 or 1   (0=appropriate, 1=inappropriate)
- confidence: float in [0.0, 1.0]
- rationale: one short sentence
- IMPORTANT: rationale must NOT contain any quote characters (no " and no ').
"""


def build_rem_stage1_prompt(
    text: str,
    ds: DatasetEnum,
    similar_texts: List[str],
) -> str:
    sims = "\n".join([f"{i+1}. {t}" for i, t in enumerate(similar_texts)])

    return f"""{_STAGE1_SYSTEM}

TEXT:
{text}

SIMILAR TEXTS (reference only):
{sims}
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


def build_rem_stage2_prompt(
    *,
    ds: DatasetEnum,
    text: str,
    similar_texts: List[str],
    true_label: int,
    stage1_pred: int,
    stage1_rationale: str,
) -> str:
    perspective = build_dataset_perspective(ds)
    sims = "\n".join([f"{i+1}. {t}" for i, t in enumerate(similar_texts)])

    is_correct = int(stage1_pred) == int(true_label)

    if is_correct:
        focus = """Case: CORRECT
Task:
- Extract the single most decisive evidence in TEXT that supports the true label.
- Write a one-line reusable rule (memory) that generalizes this decision.
Be concise."""
    else:
        focus = """Case: WRONG
Task:
- Identify what in TEXT likely triggered the wrong decision (ERROR).
- Identify what should have been prioritized/checked to reach the true label (MISSING).
- Write a one-line reusable rule (memory) that prevents this mistake next time.
Be concise."""

    return f"""You are a reflection agent.

{perspective}

true_label: {int(true_label)}
stage1_pred: {int(stage1_pred)}
stage1_rationale: {stage1_rationale}

TEXT:
{text}

SIMILAR TEXTS (reference only):
{sims}

{focus}

{_STAGE2_OUTPUT_RULES}
"""
