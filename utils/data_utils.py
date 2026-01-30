# utils/data_utils.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Union

from typing import List


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    JSONL 파일을 읽어서 dict 리스트로 반환.
    - 빈 줄은 무시
    - JSON 파싱 실패 라인은 스킵
    """
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows
