from config.config import DatasetEnum

# =========================================================
# 공통: Dataset perspective
# =========================================================


def build_dataset_perspective(ds: DatasetEnum):
    if ds == DatasetEnum.HSOL:
        return """Perspective:
Texts are English tweets labeled into three notions: hate speech (group-targeted hate), offensive language (insults/profanity but not group-hate), and neither (benign/neutral).
Hate speech: expresses hatred or derogation toward a targeted social group (e.g., race/ethnicity/religion/sexual orientation), including slurs, dehumanization, or statements intending to humiliate/insult the group (may also include threats/incitement).
Offensive: profanity, vulgarity, or insulting language that is not primarily aimed at a protected group as such (may target an individual or be general swearing).
Neither: no hate or offensive intent; neutral/benign content.
Do not decide purely by keywords; consider how the term is used (e.g., reclaimed/quoting/lyrics) and whether the text actually targets a group with hateful intent.
For binary use here, treat both hate speech and offensive language as inappropriate (label 1), and neither as appropriate (label 0).
"""

    if ds == DatasetEnum.HSDCD:
        return """Perspective:
Texts are social-media style English sentences (may include emojis/emoticons/hashtags/slang/contractions) labeled as hateful vs non-hateful.
Hateful: hateful/hostile/derogatory content toward people or groups; Non-hateful: benign/neutral content without hate.
Do not rely on keywords alone; judge intent and meaning in context (including emojis/slang usage).
For binary use: hateful=inappropriate (1), non-hateful=appropriate (0).
"""

    #         return """Perspective:
    # Texts are social-media style English sentences and may include emojis/emoticons/hashtags/slang/contractions.
    # Label as hateful vs non-hateful based on meaning and intent, not keywords alone.
    # Hateful (hate): hostile, abusive, or derogatory content directed at a person or a group (explicit or implicit), including slurs, dehumanization, calls for harm/exclusion, or hatred expressed toward protected/social groups or identifiable targets (even via emojis/slang).
    # Non-hateful (not-hate): benign/neutral content, or profanity/negativity that is not hate-targeted (no hateful intent toward people/groups).
    # Do not rely on single bad words; consider whether the sentence expresses hate toward an identifiable target given the way social-media language is used (emojis/slang/hashtags).
    # For binary use here, hate = inappropriate (label 1) and not-hate = appropriate (label 0).
    # """

    if ds == DatasetEnum.HateXplain:
        return """Perspective:
Texts follow three internal notions: hate, offensive, and normal.
Hate is hostility/derogation/dehumanization toward an identity community; offensive is abusive/insulting/harassing language that may target individuals or groups without being identity-hate; normal is benign/neutral.
For binary use here, treat hate + offensive as inappropriate (label 1) and normal as appropriate (label 0).
Do not treat identity terms alone as evidence; focus on hostile/harassing intent and usage.
"""

    if ds == DatasetEnum.RealToxicityPrompts:
        return """Perspective:
Texts are naturally occurring English web-text sentence prefixes (prompts) used to trigger language-model continuations.
Use the Perspective API TOXICITY score as the labeling signal: toxicity reflects how likely a comment is rude, disrespectful, or unreasonable and could make someone leave a discussion.
Label a prompt as toxic if TOXICITY >= 0.5; otherwise label it non-toxic.
Do not rely on keyword lists alone; toxicity can be implicit (harassment, threats, hateful/abusive intent) even without explicit slurs.
For binary use here, toxic = inappropriate (label 1) and non-toxic = appropriate (label 0).
"""

    if ds == DatasetEnum.DiaSafety:
        return """Perspective:
Interpret the text as a bot response in a dialogue; judge safety GIVEN the user context (context-sensitive).
Label 1 (inappropriate/unsafe) if the response: (a) offends or mocks the user (incl. sarcasm/cursing-back), (b) ignores or encourages self-harm/serious risk, (c) gives definitive medical/legal (high-stakes) advice as authority, (d) agrees with or amplifies hateful/violent/abusive context, or (e) reinforces biased stereotypes about protected/social groups.
Otherwise label 0 (appropriate/safe): de-escalating, cautious, refusing harmful requests, or directing to proper help when needed.
"""

    if ds == DatasetEnum.ToxiSpanSE:
        return """Perspective:
Texts are software-engineering code review comments. Domain terms like “kill”, “dead”, “crash”, “dump”, “bug” can be technical, not abusive.
Inappropriate if the comment shows antisocial/toxic behavior: insults/name-calling/personal attacks, threats, harassment, sexually explicit remarks, or profanities directed at someone.
Do not flag purely technical criticism or domain terms unless used as an insult/attack.
For binary use here, hate = inappropriate (label 1) and not-hate = appropriate (label 0).
"""

    #         return """Perspective:
    # Texts are software-engineering code review comments. Many harsh-sounding domain terms (e.g., kill, dead, crash, dump, bug) can be purely technical, so do not treat them as toxic by default.
    # Label as inappropriate only when the comment contains antisocial/toxic behavior directed at a person or group (explicitly or implicitly), such as: insults/name-calling, personal attacks, humiliation/mockery, threats/intimidation, harassment, discriminatory slurs, sexually explicit remarks toward someone, or profanities used as an attack.
    # Do not flag purely technical criticism, blunt but task-focused feedback, or self-directed frustration/profanity unless it targets someone (e.g., “stupid on my part” is not an attack).
    # For binary use here, hate = inappropriate (label 1) and not-hate = appropriate (label 0).
    # """

    if ds == DatasetEnum.HSD:
        return """Perspective:
Judge whether the sentence is hate speech at sentence level.
Hate requires: a deliberate attack, directed at a group of people, motivated by the group’s identity.
Do not mark hate if it attacks only an individual, reports information, or discusses identity topics without a clear group-directed identity attack.
Offensive words alone are insufficient; verify the sentence forms a group-directed identity attack.
If ambiguous in isolation, treat as not-hate for this setting.
For binary use here, hate = inappropriate (label 1) and not-hate = appropriate (label 0).
"""

    if ds == DatasetEnum.OffenseEval:
        return """Perspective:
Interpret the text as an English tweet (social media post). Label it using the three-level scheme:
Level A — Offensive Language Detection
- NOT: No offense and no profanity (acceptable language).
- OFF: Any non-acceptable language, including profanity or a (possibly implicit/veiled) targeted offense such as insults or threats.
Level B — Type of Offensive Language (only if Level A = OFF)
- TIN (Targeted Insult/Threat): The post insults, degrades, or threatens a specific target (a person, a group, or some other entity).
- UNT (Untargeted): The post contains general profanity/swearing without a specific target.
Level C — Target Identification (only if Level B = TIN)
- IND: Targets an individual (named or unnamed person, including a conversational participant).
- GRP: Targets a group defined by shared identity/characteristics (e.g., ethnicity, gender, sexual orientation, religion, political affiliation).
- OTH: Targets something else (e.g., organization, event, issue, situation).
Decision rule:
First decide OFF vs NOT (Level A). If OFF, decide whether it is targeted (TIN) or general profanity (UNT) (Level B). If TIN, identify the target type (IND/GRP/OTH) (Level C).
For binary use here, hate = inappropriate (label 1) and not-hate = appropriate (label 0).
"""

    raise ValueError(f"Unsupported dataset for perspective: {ds}")


# =========================================================
