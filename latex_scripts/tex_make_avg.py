from pathlib import Path
import csv
import json

# =========================
# 전역 설정
# =========================
EVAL_DIRS = [
    Path("runs_k3/test_memory/GPT5_MINI/chain_of_debate/ali_off/eval"),
    Path("runs_k3/test_memory/GPT5_MINI/chain_of_debate/ali_off/eval_1"),
    Path("runs_k3/test_memory/GPT5_MINI/chain_of_debate/ali_off/eval_2"),
]

MODEL_NAME = "GPT5_MINI"
ROW_NAME = "REM (CoD)"  # "Online (CoE)" "Online (CoD)" "Few-shot" "REM (CoE)" "REMICA (CoE)" "REM (CoD)" "REMICA (CoD)"

OUT_TEX_NAME = f"metrics_row_{ROW_NAME.replace(' ', '_')}.tex"

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
            metric_name = row["metric"]
            raw_value = row["value"]
            try:
                metrics[metric_name] = float(raw_value)
            except ValueError:
                continue

    return metrics


# =========================
# 시간 정보 로드 함수
# =========================
def load_median_ms(json_path: Path):
    if not json_path.exists():
        return None

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if "median_ms" not in data:
        return None

    try:
        return float(data["median_ms"])
    except ValueError:
        return None


# =========================
# 평균 계산 유틸
# =========================
def mean_or_none(values: list[float]):
    if len(values) == 0:
        return None
    return sum(values) / len(values)


# =========================
# LaTeX row 생성 (한 줄로)
# =========================
cells = [f"\\hspace{{0.5em}}{ROW_NAME}"]

for _, fname in DATASETS:
    # metric 평균
    for key in METRIC_KEYS:
        vals = []
        for eval_dir in EVAL_DIRS:
            csv_path = eval_dir / f"{fname}_{MODEL_NAME}_metrics.csv"
            metrics = load_metrics(csv_path)
            if key in metrics:
                vals.append(metrics[key])

        m = mean_or_none(vals)
        if m is None:
            cells.append("--")
        else:
            cells.append(f"{m * 100:.3f}")

    # time 평균
    time_vals_ms = []
    for eval_dir in EVAL_DIRS:
        time_path = eval_dir / f"{fname}_time.json"
        ms = load_median_ms(time_path)
        if ms is not None:
            time_vals_ms.append(ms)

    m_ms = mean_or_none(time_vals_ms)
    if m_ms is None:
        cells.append("--")
    else:
        sec = m_ms / 1000.0
        cells.append(f"{sec:.2f}s")

final_row = " & ".join(cells) + " \\\\\n\\hline\n"

# =========================
# 저장: 3개 경로에 동일 파일명으로 저장
# =========================
for eval_dir in EVAL_DIRS:
    out_tex = eval_dir / OUT_TEX_NAME
    out_tex.write_text(final_row, encoding="utf-8")

print("[OK] LaTeX row saved to:")
for eval_dir in EVAL_DIRS:
    print(f"  - {eval_dir / OUT_TEX_NAME}")
