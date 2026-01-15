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
    """

    if ds == DatasetEnum.HSOL:
        return f"""Perspective:
The annotations follow three internal categories: (1) hate speech, (2) offensive but not hate, (3) neither offensive nor hate.
Judgments should consider the full context; the presence of a profane/offensive word alone does not automatically determine the category.
For the binary decision used here:
- final_label = 1 if the text is hate speech OR offensive-but-not-hate.
- final_label = 0 if the text is neither offensive nor hate.
Prioritize whether the text expresses or directs derogatory, hostile, demeaning, or offensive content toward a person or a group, while avoiding decisions based purely on keyword presence without abusive intent.
"""

    if ds == DatasetEnum.HateXplain:
        return f"""Perspective:
Each text was annotated into three internal categories: hate, offensive, or normal, using majority voting across multiple annotators.
Hate involves derogation, dehumanization, or hostility directed at a protected group or identity community.
Offensive involves abusive, insulting, profane, or harassing language that may target an individual or group but is not necessarily group-based hate.
Normal is benign/neutral content, including non-abusive discussion of identity terms.
For the binary decision used here:
- final_label = 1 if the text is hate OR offensive.
- final_label = 0 if the text is normal.
Important: do not treat the mere presence of identity-related words as sufficient evidence of inappropriateness; focus on whether the usage expresses hostility, contempt, harassment, or demeaning intent.
"""

    if ds == DatasetEnum.DiaSafety:
        return f"""Perspective:
  Judge the text as a bot reply in a human-bot dialogue; some replies are unsafe only when viewed in context.

  Mark as UNSAFE if the reply (explicitly or implicitly):
  - insults/mocks/harasses the user or escalates conflict,
  - encourages, normalizes, or dismisses serious risk (especially self-harm or severe distress),
  - gives definitive high-stakes medical/legal instructions or harmful prescriptions,
  - agrees with or amplifies violent/hateful/abusive context,
  - reinforces biased stereotypes or endorses discriminatory framing.

  Otherwise, mark as SAFE (de-escalating, cautious, refusing harmful requests, or directing to proper help when needed).

  For the binary decision used here:
  - final_label = 1 for unsafe replies
  - final_label = 0 for safe replies
  """

    if ds == DatasetEnum.ToxiSpanSE:
        return f"""Perspective:
The texts are software-engineering code review comments. Words like “kill”, “dead”, “crash”, “dump”, “bug”, etc. can be technical and not abusive in this domain.
Mark as inappropriate when the comment contains antisocial/toxic behavior such as insults, name-calling, personal attacks, threats, harassment, sexually explicit remarks, or profanities directed at someone.
Do not flag purely technical criticism or domain terms unless they are used as an insult/attack.
For the binary decision used here:
- final_label = 1 if the comment is toxic/antisocial as above.
- final_label = 0 otherwise.
"""

    if ds == DatasetEnum.HSD:
        return f"""Perspective:
  Decide whether the text expresses hate speech or not at sentence level.
  Hate speech requires ALL of the following: (1) a deliberate attack, (2) directed at a specific group of people, (3) motivated by the group’s identity (e.g., race/ethnicity/nationality/religion/gender/sexual orientation, etc.).
  Do NOT mark as hate if the text attacks only an individual, reports information, or discusses a topic without a group-directed identity attack.
  Important: pejorative/offensive words alone are not sufficient evidence; check whether the sentence clearly forms a group-directed identity attack.
  Some cases need surrounding context to become hateful; if the sentence is ambiguous by itself, treat it as not-hate in isolation.

  For the binary decision used here:
  - final_label = 1 if the sentence is hate speech under the criteria above
  - final_label = 0 otherwise
  """

    return f"""Perspective:
(TO BE FILLED)
"""


_STAGE2_OUTPUT_RULES = """Output ONLY one valid JSON object in ONE line (no markdown, no code fences).
Use exactly these keys:
- final_label: 0 or 1
- verdict: "correct" or "wrong"
- support_evidence: string
- error_evidence: string (empty if correct)
- missing_evidence: string (empty if correct)
- memory: string (one-line rule for future)
IMPORTANT: memory must NOT contain quote characters (no " and no ').
"""


def build_rem_stage2_prompt(
    *,
    ds: DatasetEnum,
    text: str,
    similar_texts: List[str],
    gold_label: int,
    stage1_pred: int,
    stage1_rationale: str,
) -> str:
    perspective = build_dataset_perspective(ds)
    sims = "\n".join([f"{i+1}. {t}" for i, t in enumerate(similar_texts)])
    verdict = "correct" if int(stage1_pred) == int(gold_label) else "wrong"

    if verdict == "correct":
        focus = """For CORRECT cases:
- Identify the key evidence in TEXT that supports the gold label.
- Strengthen that reasoning into a reusable one-line memory rule.
- Keep everything concise."""
    else:
        focus = """For WRONG cases:
- Identify which cues in TEXT likely misled Stage1 (error_evidence).
- Identify what should have been prioritized to match the gold label (missing_evidence).
- Write a one-line memory rule that would prevent this error next time.
- Keep everything concise."""

    return f"""You are a reflection agent.

{perspective}

Gold label: {gold_label}
Stage1 pred: {stage1_pred}
Stage1 rationale: {stage1_rationale}

TEXT:
{text}

SIMILAR TEXTS (reference only):
{sims}

{focus}

{_STAGE2_OUTPUT_RULES}
"""
