import re


def is_only_url_or_symbol(text: str) -> bool:
    """텍스트가 순수한 URL, 해시태그, 멘션으로만 구성되었는지 판별"""
    cleaned = text.strip()

    # 완전한 공백 or 줄바꿈만 있는 경우
    if not cleaned or cleaned == "\u2028" or cleaned == "\u2029":
        return True

    # 모든 문자를 제거한 뒤 비어 있으면 의미 없는 텍스트
    if len(re.sub(r"[^\w]", "", cleaned)) == 0:
        return True

    # URL만 있는 경우
    if re.fullmatch(r"(https?://\S+)", cleaned):
        return True

    # 여러 URL만 있을 경우
    if re.fullmatch(r"(https?://\S+\s*)+", cleaned):
        return True

    # 해시태그나 멘션만 있는 경우
    if re.fullmatch(r"([#@]\w+\s*)+", cleaned):
        return True

    return False
