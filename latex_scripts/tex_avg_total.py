from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json


# =========================
# 사용자 설정
# =========================

RUN_ROOTS = [
    Path("runs_k3_1"),
    Path("runs_k3_2"),
    Path("runs_k3_3"),
]

OUT_DIR_PREFIX = "runs_k3__"
OUT_DIR_SUFFIX = "_avg"
OUT_TEX_NAME = "metrics_table_body_avg.tex"
OUT_SELECTION_NAME = "eval_selection.json"
OUT_SUMMARY_NAME = "summary.json"

DATASETS = [
    ("DiaSafety", "DiaSafety"),
    ("HSDCD", "HSDCD"),
    ("RTP", "RealToxicityPrompts"),
    ("ToxiSpanSE", "ToxiSpanSE"),
]

MODEL_DISPLAY = {
    "GPT5_1": "GPT-5.1",
    "GPT5_MINI": "GPT-5.1-mini",
    "CLAUDE_HAIKU4_5": "Claude-Haiku-4.5",
}

ROW_SPECS = [
    ("Few-shot (RAG-only)", "test_memory", "none", "ali_off"),

    ("Online (CoE)", "online_test", "chain_of_expert", None),
    ("Online (CoD)", "online_test", "chain_of_debate", None),

    ("REM (CoE)", "test_memory", "chain_of_expert", "ali_off"),
    ("REM (CoD)", "test_memory", "chain_of_debate", "ali_off"),

    ("REMICA (CoE)", "test_memory", "chain_of_expert", "ali_on"),
    ("REMICA (CoD)", "test_memory", "chain_of_debate", "ali_on"),
]

MODEL_ORDER = ["GPT5_1", "GPT5_MINI", "CLAUDE_HAIKU4_5"]

# 렌더링 컬럼 개수 (Row label 1 + 4 datasets * 3 + Average * 3 = 16)
TABLE_COLS = 16


# =========================
# 데이터 구조
# =========================

@dataclass
class MetricTriple:
    acc: float | None
    f1: float | None
    time_sec: float | None


# =========================
# 기본 IO
# =========================

def load_metrics_csv(csv_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not csv_path.exists():
        return metrics

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_name = row["metric"]
            metric_value = row["value"]
            try:
                metrics[metric_name] = float(metric_value)
            except ValueError:
                continue

    return metrics


def load_median_time_sec(json_path: Path) -> float | None:
    if not json_path.exists():
        return None

    try:
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)

        if "median_ms" not in data:
            return None

        ms = float(data["median_ms"])
        return ms / 1000.0
    except Exception:
        return None


def mean(values: list[float]) -> float | None:
    if len(values) == 0:
        return None
    return sum(values) / len(values)


def format_percent(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "--"
    return f"{v * 100:.{digits}f}"


def format_time(v_sec: float | None, digits: int = 2) -> str:
    if v_sec is None:
        return "--"
    return f"{v_sec:.{digits}f}s"


def wrap_best(s: str) -> str:
    return f"\\textbf{{{s}}}"


def wrap_second(s: str) -> str:
    return f"\\underline{{{s}}}"


# =========================
# 출력 폴더명
# =========================

def extract_run_suffix(run_root: Path) -> str:
    name = run_root.name
    if "_" not in name:
        return name
    return name.split("_")[-1]


def build_out_dir_name(run_roots: list[Path]) -> str:
    suffixes: list[str] = []
    for rr in run_roots:
        suffixes.append(extract_run_suffix(rr))
    joined = "_".join(suffixes)
    return f"{OUT_DIR_PREFIX}{joined}{OUT_DIR_SUFFIX}"


# =========================
# eval 후보 탐지 + 선택
# =========================

def list_eval_candidates(eval_parent: Path) -> list[Path]:
    """
    eval_parent 아래에서 eval, eval1, eval2 ... 형태의 디렉토리를 후보로 수집.
    """
    if not eval_parent.exists():
        return []

    candidates: list[Path] = []
    for p in eval_parent.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name == "eval":
            candidates.append(p)
        elif name.startswith("eval"):
            suffix = name[4:]
            if suffix.isdigit():
                candidates.append(p)

    def sort_key(x: Path):
        if x.name == "eval":
            return (0, 0)
        return (1, int(x.name[4:]))

    candidates.sort(key=sort_key)
    return candidates


def make_selection_key(run_root: Path, mode: str, model_name: str, chain: str, ali_mode: str | None) -> str:
    if ali_mode is None:
        return f"{run_root.as_posix()}|{mode}|{model_name}|{chain}"
    return f"{run_root.as_posix()}|{mode}|{model_name}|{chain}|{ali_mode}"


def choose_one_eval_dir(
    selection_map: dict[str, str],
    run_root: Path,
    mode: str,
    model_name: str,
    chain: str,
    ali_mode: str | None,
) -> Path | None:
    if mode == "online_test":
        eval_parent = run_root / "online_test" / model_name / chain
    else:
        if chain == "none":
            if ali_mode is None:
                return None
            eval_parent = run_root / "test_memory" / model_name / "none" / ali_mode
        else:
            if ali_mode is None:
                return None
            eval_parent = run_root / "test_memory" / model_name / chain / ali_mode

    candidates = list_eval_candidates(eval_parent)
    if len(candidates) == 0:
        return None

    if len(candidates) == 1:
        return candidates[0]

    sel_key = make_selection_key(run_root, mode, model_name, chain, ali_mode)
    if sel_key in selection_map:
        chosen_name = selection_map[sel_key]
        for c in candidates:
            if c.name == chosen_name:
                return c

    print("")
    print("여러 eval 후보가 발견되었습니다. 사용할 eval을 선택하세요.")
    print(f"- run_root: {run_root}")
    print(f"- mode: {mode}, model: {model_name}, chain: {chain}, ali: {ali_mode}")
    print(f"- parent: {eval_parent}")
    print("후보 목록:")
    for i, c in enumerate(candidates, start=1):
        print(f"  {i}) {c.name}")

    while True:
        s = input("번호를 입력하세요: ").strip()
        if not s.isdigit():
            print("숫자를 입력해 주세요.")
            continue
        idx = int(s)
        if idx < 1 or idx > len(candidates):
            print("범위를 벗어났습니다.")
            continue
        chosen = candidates[idx - 1]
        selection_map[sel_key] = chosen.name
        return chosen


def load_selection_map(selection_path: Path) -> dict[str, str]:
    if not selection_path.exists():
        return {}
    try:
        with selection_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        return {}


def save_selection_map(selection_path: Path, selection_map: dict[str, str]):
    selection_path.write_text(json.dumps(selection_map, indent=2, ensure_ascii=False), encoding="utf-8")


# =========================
# metrics 수집/집계
# =========================

def collect_one_eval(eval_dir: Path, model_name: str) -> dict[str, MetricTriple]:
    out: dict[str, MetricTriple] = {}

    for ds_key, fname in DATASETS:
        csv_path = eval_dir / f"{fname}_{model_name}_metrics.csv"
        metrics = load_metrics_csv(csv_path)

        acc_val = metrics["accuracy"] if "accuracy" in metrics else None
        f1_val = metrics["f1"] if "f1" in metrics else None

        time_path = eval_dir / f"{fname}_time.json"
        time_sec = load_median_time_sec(time_path)

        out[ds_key] = MetricTriple(acc=acc_val, f1=f1_val, time_sec=time_sec)

    return out


def aggregate_across_runs(
    run_roots: list[Path],
    selection_map: dict[str, str],
    mode: str,
    model_name: str,
    chain: str,
    ali_mode: str | None,
) -> dict[str, MetricTriple]:
    per_dataset_acc: dict[str, list[float]] = {}
    per_dataset_f1: dict[str, list[float]] = {}
    per_dataset_time: dict[str, list[float]] = {}

    for ds_key, _ in DATASETS:
        per_dataset_acc[ds_key] = []
        per_dataset_f1[ds_key] = []
        per_dataset_time[ds_key] = []

    for run_root in run_roots:
        eval_dir = choose_one_eval_dir(selection_map, run_root, mode, model_name, chain, ali_mode)
        if eval_dir is None:
            continue

        metrics_map = collect_one_eval(eval_dir, model_name)

        for ds_key in metrics_map:
            triple = metrics_map[ds_key]
            if triple.acc is not None:
                per_dataset_acc[ds_key].append(triple.acc)
            if triple.f1 is not None:
                per_dataset_f1[ds_key].append(triple.f1)
            if triple.time_sec is not None:
                per_dataset_time[ds_key].append(triple.time_sec)

    out: dict[str, MetricTriple] = {}
    for ds_key in per_dataset_acc:
        out[ds_key] = MetricTriple(
            acc=mean(per_dataset_acc[ds_key]),
            f1=mean(per_dataset_f1[ds_key]),
            time_sec=mean(per_dataset_time[ds_key]),
        )

    return out


def compute_average_over_datasets(ds_map: dict[str, MetricTriple]) -> MetricTriple:
    acc_list: list[float] = []
    f1_list: list[float] = []
    time_list: list[float] = []

    for ds_key in ds_map:
        triple = ds_map[ds_key]
        if triple.acc is not None:
            acc_list.append(triple.acc)
        if triple.f1 is not None:
            f1_list.append(triple.f1)
        if triple.time_sec is not None:
            time_list.append(triple.time_sec)

    return MetricTriple(
        acc=mean(acc_list),
        f1=mean(f1_list),
        time_sec=mean(time_list),
    )


# =========================
# 최고/2등 계산
# =========================

def rank_best_second(values_by_row: dict[str, float], higher_better: bool) -> tuple[set[str], set[str]]:
    """
    values_by_row: row_name -> value (None 제외하고 넣기)
    higher_better: True면 내림차순(큰 값이 좋음), False면 오름차순(작은 값이 좋음)
    동점은 모두 같은 랭크 처리.
    """
    eps = 1e-12

    items = [(r, v) for r, v in values_by_row.items()]
    if len(items) == 0:
        return set(), set()

    items.sort(key=lambda x: x[1], reverse=higher_better)

    best_value = items[0][1]
    best_rows: set[str] = set()
    for r, v in items:
        if abs(v - best_value) <= eps:
            best_rows.add(r)

    second_value = None
    for _, v in items:
        if abs(v - best_value) > eps:
            second_value = v
            break

    second_rows: set[str] = set()
    if second_value is not None:
        for r, v in items:
            if abs(v - second_value) <= eps:
                second_rows.add(r)

    return best_rows, second_rows


def build_highlight_maps_for_model(row_results: dict[str, dict[str, MetricTriple]]):
    """
    row_results:
      row_name -> ds_key -> MetricTriple

    반환:
      highlights[(row_name, ds_key, metric)] = "best" | "second" | None
      highlights[(row_name, "__avg__", metric)] 도 포함 (Average 컬럼)
    """
    highlights: dict[tuple[str, str, str], str] = {}

    # per dataset
    for ds_key, _ in DATASETS:
        acc_vals: dict[str, float] = {}
        f1_vals: dict[str, float] = {}
        time_vals: dict[str, float] = {}

        for row_name in row_results:
            triple = row_results[row_name][ds_key]
            if triple.acc is not None:
                acc_vals[row_name] = triple.acc
            if triple.f1 is not None:
                f1_vals[row_name] = triple.f1
            if triple.time_sec is not None:
                time_vals[row_name] = triple.time_sec

        best, second = rank_best_second(acc_vals, higher_better=True)
        for r in best:
            highlights[(r, ds_key, "acc")] = "best"
        for r in second:
            highlights[(r, ds_key, "acc")] = "second"

        best, second = rank_best_second(f1_vals, higher_better=True)
        for r in best:
            highlights[(r, ds_key, "f1")] = "best"
        for r in second:
            highlights[(r, ds_key, "f1")] = "second"

        best, second = rank_best_second(time_vals, higher_better=False)
        for r in best:
            highlights[(r, ds_key, "time")] = "best"
        for r in second:
            highlights[(r, ds_key, "time")] = "second"

    # average column
    avg_acc_vals: dict[str, float] = {}
    avg_f1_vals: dict[str, float] = {}
    avg_time_vals: dict[str, float] = {}

    for row_name in row_results:
        avg_triple = compute_average_over_datasets(row_results[row_name])
        if avg_triple.acc is not None:
            avg_acc_vals[row_name] = avg_triple.acc
        if avg_triple.f1 is not None:
            avg_f1_vals[row_name] = avg_triple.f1
        if avg_triple.time_sec is not None:
            avg_time_vals[row_name] = avg_triple.time_sec

    best, second = rank_best_second(avg_acc_vals, higher_better=True)
    for r in best:
        highlights[(r, "__avg__", "acc")] = "best"
    for r in second:
        highlights[(r, "__avg__", "acc")] = "second"

    best, second = rank_best_second(avg_f1_vals, higher_better=True)
    for r in best:
        highlights[(r, "__avg__", "f1")] = "best"
    for r in second:
        highlights[(r, "__avg__", "f1")] = "second"

    best, second = rank_best_second(avg_time_vals, higher_better=False)
    for r in best:
        highlights[(r, "__avg__", "time")] = "best"
    for r in second:
        highlights[(r, "__avg__", "time")] = "second"

    return highlights


# =========================
# LaTeX 렌더링
# =========================

def render_group_header(model_display_name: str) -> str:
    lines: list[str] = []
    lines.append("\\Xhline{1.0pt}")
    lines.append("% ---------- Body: group header ----------")
    lines.append(f"\\multicolumn{{{TABLE_COLS}}}{{l}}{{\\textbf{{{model_display_name}}}}} \\\\")
    lines.append("\\Xhline{1.0pt}")
    lines.append("")
    return "\n".join(lines)


def apply_highlight(row_name: str, ds_key: str, metric: str, s: str, highlights: dict[tuple[str, str, str], str]) -> str:
    k = (row_name, ds_key, metric)
    if k not in highlights:
        return s
    if highlights[k] == "best":
        return wrap_best(s)
    if highlights[k] == "second":
        return wrap_second(s)
    return s


def render_row(row_name: str, ds_map: dict[str, MetricTriple], highlights: dict[tuple[str, str, str], str]) -> str:
    parts: list[str] = []
    parts.append(f"\\hspace{{0.5em}}{row_name}")

    for ds_key, _ in DATASETS:
        triple = ds_map[ds_key]

        acc_s = format_percent(triple.acc)
        f1_s = format_percent(triple.f1)
        time_s = format_time(triple.time_sec)

        acc_s = apply_highlight(row_name, ds_key, "acc", acc_s, highlights)
        f1_s = apply_highlight(row_name, ds_key, "f1", f1_s, highlights)
        time_s = apply_highlight(row_name, ds_key, "time", time_s, highlights)

        parts.append(f"   & {acc_s} & {f1_s} & {time_s}")

    avg_triple = compute_average_over_datasets(ds_map)

    avg_acc_s = format_percent(avg_triple.acc)
    avg_f1_s = format_percent(avg_triple.f1)
    avg_time_s = format_time(avg_triple.time_sec)

    avg_acc_s = apply_highlight(row_name, "__avg__", "acc", avg_acc_s, highlights)
    avg_f1_s = apply_highlight(row_name, "__avg__", "f1", avg_f1_s, highlights)
    avg_time_s = apply_highlight(row_name, "__avg__", "time", avg_time_s, highlights)

    parts.append(f"   & {avg_acc_s} & {avg_f1_s} & {avg_time_s} \\\\")
    parts.append("\\hline")
    parts.append("")
    return "\n".join(parts)


# =========================
# summary.json 생성
# =========================

def pick_best_second_avg(row_avg_values: dict[str, float], higher_better: bool):
    best_rows, second_rows = rank_best_second(row_avg_values, higher_better=higher_better)

    def get_value(row_name: str):
        return row_avg_values[row_name]

    best_list = sorted(list(best_rows), key=get_value, reverse=higher_better)
    second_list = sorted(list(second_rows), key=get_value, reverse=higher_better)

    best_item = None
    second_item = None

    if len(best_list) > 0:
        r = best_list[0]
        best_item = {"row": r, "score": row_avg_values[r]}
    if len(second_list) > 0:
        r = second_list[0]
        second_item = {"row": r, "score": row_avg_values[r]}

    return best_item, second_item


def to_percent_3(v: float | None) -> float | None:
    if v is None:
        return None
    return round(v * 100.0, 3)


def to_sec_2(v: float | None) -> float | None:
    if v is None:
        return None
    return round(v, 2)


def build_summary_for_model(row_results: dict[str, dict[str, MetricTriple]]):
    per_dataset_summary: dict[str, dict[str, dict[str, object]]] = {}

    # per dataset best/second for each metric
    for ds_key, _ in DATASETS:
        acc_vals: dict[str, float] = {}
        f1_vals: dict[str, float] = {}
        time_vals: dict[str, float] = {}

        for row_name in row_results:
            triple = row_results[row_name][ds_key]
            if triple.acc is not None:
                acc_vals[row_name] = triple.acc
            if triple.f1 is not None:
                f1_vals[row_name] = triple.f1
            if triple.time_sec is not None:
                time_vals[row_name] = triple.time_sec

        best, second = pick_best_second_avg(acc_vals, higher_better=True)
        acc_block = {"best": best, "second": second}
        if acc_block["best"] is not None:
            acc_block["best"]["score"] = to_percent_3(acc_block["best"]["score"])
        if acc_block["second"] is not None:
            acc_block["second"]["score"] = to_percent_3(acc_block["second"]["score"])

        best, second = pick_best_second_avg(f1_vals, higher_better=True)
        f1_block = {"best": best, "second": second}
        if f1_block["best"] is not None:
            f1_block["best"]["score"] = to_percent_3(f1_block["best"]["score"])
        if f1_block["second"] is not None:
            f1_block["second"]["score"] = to_percent_3(f1_block["second"]["score"])

        best, second = pick_best_second_avg(time_vals, higher_better=False)
        time_block = {"best": best, "second": second}
        if time_block["best"] is not None:
            time_block["best"]["score"] = to_sec_2(time_block["best"]["score"])
        if time_block["second"] is not None:
            time_block["second"]["score"] = to_sec_2(time_block["second"]["score"])

        per_dataset_summary[ds_key] = {
            "accuracy": acc_block,
            "f1": f1_block,
            "time_sec": time_block,
        }

    # average over datasets per row
    avg_acc_vals: dict[str, float] = {}
    avg_f1_vals: dict[str, float] = {}
    avg_time_vals: dict[str, float] = {}

    for row_name in row_results:
        avg_triple = compute_average_over_datasets(row_results[row_name])
        if avg_triple.acc is not None:
            avg_acc_vals[row_name] = avg_triple.acc
        if avg_triple.f1 is not None:
            avg_f1_vals[row_name] = avg_triple.f1
        if avg_triple.time_sec is not None:
            avg_time_vals[row_name] = avg_triple.time_sec

    best, second = pick_best_second_avg(avg_acc_vals, higher_better=True)
    avg_accuracy = {"best": best, "second": second}
    if avg_accuracy["best"] is not None:
        avg_accuracy["best"]["score"] = to_percent_3(avg_accuracy["best"]["score"])
    if avg_accuracy["second"] is not None:
        avg_accuracy["second"]["score"] = to_percent_3(avg_accuracy["second"]["score"])

    best, second = pick_best_second_avg(avg_f1_vals, higher_better=True)
    avg_f1 = {"best": best, "second": second}
    if avg_f1["best"] is not None:
        avg_f1["best"]["score"] = to_percent_3(avg_f1["best"]["score"])
    if avg_f1["second"] is not None:
        avg_f1["second"]["score"] = to_percent_3(avg_f1["second"]["score"])

    best, second = pick_best_second_avg(avg_time_vals, higher_better=False)
    avg_time = {"best": best, "second": second}
    if avg_time["best"] is not None:
        avg_time["best"]["score"] = to_sec_2(avg_time["best"]["score"])
    if avg_time["second"] is not None:
        avg_time["second"]["score"] = to_sec_2(avg_time["second"]["score"])

    return {
        "avg_accuracy": avg_accuracy,
        "avg_f1": avg_f1,
        "avg_time_sec": avg_time,
        "per_dataset": per_dataset_summary,
    }


# =========================
# 테이블 바디 빌드
# =========================

def build_table_body_and_summary(run_roots: list[Path], selection_map: dict[str, str]):
    lines: list[str] = []
    summary_models: dict[str, object] = {}

    for model_name in MODEL_ORDER:
        display_name = MODEL_DISPLAY[model_name] if model_name in MODEL_DISPLAY else model_name

        # 1) 먼저 row 결과를 전부 모아둠
        row_results: dict[str, dict[str, MetricTriple]] = {}
        for row_name, mode, chain, ali_mode in ROW_SPECS:
            ds_map = aggregate_across_runs(
                run_roots=run_roots,
                selection_map=selection_map,
                mode=mode,
                model_name=model_name,
                chain=chain,
                ali_mode=ali_mode,
            )
            row_results[row_name] = ds_map

        # 2) 하이라이트 맵 생성
        highlights = build_highlight_maps_for_model(row_results)

        # 3) 렌더
        lines.append(render_group_header(display_name))
        for row_name, _, _, _ in ROW_SPECS:
            lines.append(render_row(row_name, row_results[row_name], highlights))

        # 4) summary 생성
        summary_models[model_name] = {
            "display_name": display_name,
            "summary": build_summary_for_model(row_results),
        }

    body = "\n".join(lines).rstrip() + "\n"
    summary = {"models": summary_models}
    return body, summary


# =========================
# Main
# =========================

def main():
    out_dir = Path(build_out_dir_name(RUN_ROOTS))
    out_dir.mkdir(parents=True, exist_ok=True)

    selection_path = out_dir / OUT_SELECTION_NAME
    selection_map = load_selection_map(selection_path)

    tex_body, summary = build_table_body_and_summary(RUN_ROOTS, selection_map)

    save_selection_map(selection_path, selection_map)

    out_tex = out_dir / OUT_TEX_NAME
    out_tex.write_text(tex_body, encoding="utf-8")

    out_summary = out_dir / OUT_SUMMARY_NAME
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    print(f"[OK] output dir: {out_dir}")
    print(f"[OK] saved selection: {selection_path}")
    print(f"[OK] saved tex: {out_tex}")
    print(f"[OK] saved summary: {out_summary}")


if __name__ == "__main__":
    main()
