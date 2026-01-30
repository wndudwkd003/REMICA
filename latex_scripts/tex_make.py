from pathlib import Path
import csv

# =========================
# 전역 설정
# =========================
EVAL_DIR = Path("runs/test_memory/CLAUDE_HAIKU4_5/chain_of_debate/ali_on/eval")
MODEL_NAME = "CLAUDE_HAIKU4_5"
ROW_NAME = "REMICA (CoD)"  # Base / Few-shot / REMICA (ours)

OUT_TEX = EVAL_DIR / f"metrics_row_{ROW_NAME.replace(' ', '_')}.tex"

DATASETS = [
    ("DiaSafety", "DiaSafety"),
    # ("HateXplain", "HateXplain"),
    ("HSDCD", "HSDCD"),
    ("HSOL", "HSOL"),
    # ("OffenseEval", "OffenseEval"),
    ("RTP", "RealToxicityPrompts"),
    ("ToxiSpanSE", "ToxiSpanSE"),
]

# accuracy와 f1만 사용
METRIC_KEYS = ["accuracy", "f1"]


# =========================
# CSV 읽기 함수
# =========================
def load_metrics(csv_path: Path) -> dict:
    metrics = {}
    if not csv_path.exists():
        return metrics

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["metric"]] = row["value"]
    return metrics


# =========================
# LaTeX row 생성
# =========================
lines = [f"\\hspace{{0.5em}}{ROW_NAME}"]

for _, fname in DATASETS:
    # ⬇⬇⬇ 모델 이름에 맞게 동적으로 파일명 구성
    csv_path = EVAL_DIR / f"{fname}_{MODEL_NAME}_metrics.csv"
    metrics = load_metrics(csv_path)

    row_part = []
    for key in METRIC_KEYS:
        v = metrics.get(key)
        if v is None:
            row_part.append("--")
        else:
            try:
                val_percent = float(v) * 100
                row_part.append(f"{val_percent:.3f}")
            except ValueError:
                row_part.append("--")
    lines.append(
        "  " + " & ".join([""] + row_part)
    )  # 앞에 & 붙이기 위해 빈 문자열 추가

# 마지막 줄은 전체 종료
lines[-1] += " \\\\"
lines.append("\\hline")

final_row = "\n".join(lines)

# =========================
# 저장
# =========================
OUT_TEX.write_text(final_row, encoding="utf-8")
print(f"[OK] LaTeX row saved to: {OUT_TEX}")
