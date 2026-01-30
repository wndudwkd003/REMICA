# utils/time_utils.py

from __future__ import annotations

import time


def now_ms() -> float:
    """
    고해상도 타이머 기반 현재 시각(ms)을 반환.
    - 상대적인 경과 시간 측정용 (절대 wall time 용도 아님).
    """
    return time.perf_counter() * 1000.0
