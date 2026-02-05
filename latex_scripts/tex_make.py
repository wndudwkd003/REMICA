from pathlib import Path
import csv
import json

# =========================
# 전역 설정
# =========================
EVAL_DIR = Path("runs/test_memory/GPT5_MINI/chain_of_debate/ali_off/eval")
MODEL_NAME = "GPT5_MINI"
ROW_NAME = "REM (CoD)"  # "Online (CoE)" "Online (CoD)" "Few-shot" "REM (CoE)" "REMICA (CoE)" "REM (CoD)" "REMICA (CoD)"

OUT_TEX = EVAL_DIR / f"metrics_row_{ROW_NAME.replace(' ', '_')}.tex"

DATASETS = [
    ("DiaSafety", "DiaSafety"),
    ("HSDCD", "HSDCD"),
    ("RTP", "RealToxicityPrompts"),
    ("ToxiSpanSE", "ToxiSpanSE"),
]

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
# 시간 정보 로드 함수
# =========================
def load_median_time(json_path: Path) -> str:
    if not json_path.exists():
        return "--"
    try:
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
        ms = float(data.get("median_ms", 0))
        sec = ms / 1000
        return f"{sec:.2f}s"
    except Exception:
        return "--"


# =========================
# LaTeX row 생성
# =========================
lines = [f"\\hspace{{0.5em}}{ROW_NAME}"]

for _, fname in DATASETS:
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

    # 중앙 응답 시간 추가
    time_path = EVAL_DIR / f"{fname}_time.json"
    median_time = load_median_time(time_path)
    row_part.append(median_time)

    lines.append("  " + " & ".join([""] + row_part))

# 마지막 줄
lines[-1] += " \\\\"
lines.append("\\hline")

final_row = "\n".join(lines)

# =========================
# 저장
# =========================
OUT_TEX.write_text(final_row, encoding="utf-8")
print(f"[OK] LaTeX row saved to: {OUT_TEX}")
