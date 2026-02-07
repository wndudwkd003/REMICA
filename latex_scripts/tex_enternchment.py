from pathlib import Path
import json


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt3(x):
    return f"{float(x):.3f}"


def model_display_name(model_key: str):
    if model_key == "GPT5_1":
        return "GPT-5.1"
    if model_key == "GPT5_MINI":
        return "GPT-5.1-mini"
    if model_key == "CLAUDE_HAIKU4_5":
        return "Claude-Haiku-4.5"
    return model_key


def chain_display_name(chain_key: str):
    if chain_key == "chain_of_expert":
        return "CoE"
    if chain_key == "chain_of_debate":
        return "CoD"
    return chain_key


def get_medians(summary_results, model_key: str, chain_key: str, agent_key: str, metric: str):
    block = summary_results[model_key][chain_key][agent_key][metric]
    off_med = block["off"]["median"]
    on_med = block["on"]["median"]
    return off_med, on_med


def apply_rank_style(text_value: str, style: str):
    if style == "bold":
        return r"\textbf{" + text_value + r"}"
    if style == "underline":
        return r"\underline{" + text_value + r"}"
    return text_value


def compute_rank_styles(values_by_row):
    pairs = []
    for row_id, v in values_by_row.items():
        pairs.append((row_id, float(v)))

    if not pairs:
        return {}

    max_val = max(v for _, v in pairs)

    distinct_vals_desc = sorted({v for _, v in pairs}, reverse=True)
    second_val = None
    if len(distinct_vals_desc) >= 2:
        second_val = distinct_vals_desc[1]

    out = {}
    for row_id, v in pairs:
        if v == max_val:
            out[row_id] = "bold"
        elif second_val is not None and v == second_val:
            out[row_id] = "underline"
        else:
            out[row_id] = ""
    return out


def build_table_tex(summary_obj):
    results = summary_obj["results"]

    model_order = ["GPT5_1", "GPT5_MINI", "CLAUDE_HAIKU4_5"]

    # REM (CoE) -> REM (CoD) -> REMICA (CoE) -> REMICA (CoD)
    row_order = [
        ("REM", "chain_of_expert"),
        ("REM", "chain_of_debate"),
        ("REMICA", "chain_of_expert"),
        ("REMICA", "chain_of_debate"),
    ]

    # 지표 이름
    metric_seq_name = r"$\Delta_{\mathrm{seq}}$"
    metric_jac_name = r"$\Delta_{\mathrm{jac}}$"
    metric_emb_name = r"$\Delta_{\mathrm{emb}}$"

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{Rationale round-change summary under REM (w/o ICA) vs REMICA (w ICA).}")
    lines.append(r"\label{tab:enternchment_summary}")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    # Model + Agent A(3) + Agent B(3) = 7 columns
    lines.append(r"\begin{tabular}{l c c c c c c}")

    # 상단 굵은 라인
    lines.append(r"\Xhline{1.0pt}")

    # 헤더 1행: Agent 그룹
    lines.append(
        r"Model & \multicolumn{3}{c}{Agent A} & \multicolumn{3}{c}{Agent B} \\"
    )

    # 그룹별 밑줄
    lines.append(r"\cline{2-4}\cline{5-7}")

    # 헤더 2행: 각 에이전트 아래 지표
    lines.append(
        r" & "
        + metric_seq_name
        + r" & "
        + metric_jac_name
        + r" & "
        + metric_emb_name
        + r" & "
        + metric_seq_name
        + r" & "
        + metric_jac_name
        + r" & "
        + metric_emb_name
        + r" \\"
    )

    # 헤더 아래 굵은 라인
    lines.append(r"\Xhline{1.0pt}")

    for model_key in model_order:
        if model_key not in results:
            continue

        mname = model_display_name(model_key)

        # row_order 순서대로 rows 생성
        rows = []
        for method_key, chain_key in row_order:
            if chain_key not in results[model_key]:
                continue

            # agent A
            a_seq_off, a_seq_on = get_medians(results, model_key, chain_key, "agent_A", "seq")
            a_jac_off, a_jac_on = get_medians(results, model_key, chain_key, "agent_A", "jac")
            a_emb_off, a_emb_on = get_medians(results, model_key, chain_key, "agent_A", "emb")

            # agent B
            b_seq_off, b_seq_on = get_medians(results, model_key, chain_key, "agent_B", "seq")
            b_jac_off, b_jac_on = get_medians(results, model_key, chain_key, "agent_B", "jac")
            b_emb_off, b_emb_on = get_medians(results, model_key, chain_key, "agent_B", "emb")

            if method_key == "REM":
                a_seq = a_seq_off
                a_jac = a_jac_off
                a_emb = a_emb_off
                b_seq = b_seq_off
                b_jac = b_jac_off
                b_emb = b_emb_off
            else:
                a_seq = a_seq_on
                a_jac = a_jac_on
                a_emb = a_emb_on
                b_seq = b_seq_on
                b_jac = b_jac_on
                b_emb = b_emb_on

            row_id = method_key + "|" + chain_key
            rows.append(
                {
                    "row_id": row_id,
                    "method_key": method_key,
                    "chain_key": chain_key,
                    "a_seq": a_seq,
                    "a_jac": a_jac,
                    "a_emb": a_emb,
                    "b_seq": b_seq,
                    "b_jac": b_jac,
                    "b_emb": b_emb,
                }
            )

        if not rows:
            continue

        # 모델 블록 내(4행)에서만 best/second 스타일 계산
        values_a_seq = {r["row_id"]: r["a_seq"] for r in rows}
        values_a_jac = {r["row_id"]: r["a_jac"] for r in rows}
        values_a_emb = {r["row_id"]: r["a_emb"] for r in rows}

        values_b_seq = {r["row_id"]: r["b_seq"] for r in rows}
        values_b_jac = {r["row_id"]: r["b_jac"] for r in rows}
        values_b_emb = {r["row_id"]: r["b_emb"] for r in rows}

        style_a_seq = compute_rank_styles(values_a_seq)
        style_a_jac = compute_rank_styles(values_a_jac)
        style_a_emb = compute_rank_styles(values_a_emb)

        style_b_seq = compute_rank_styles(values_b_seq)
        style_b_jac = compute_rank_styles(values_b_jac)
        style_b_emb = compute_rank_styles(values_b_emb)

        # 모델 헤더 행
        lines.append(r"\multicolumn{7}{l}{\textbf{" + mname + r"}} \\")
        lines.append(r"\Xhline{1.0pt}")

        # rows를 만든 순서(=REM,REM,REMICA,REMICA) 그대로 출력
        for r in rows:
            co_name = chain_display_name(r["chain_key"])

            a_seq_s = apply_rank_style(fmt3(r["a_seq"]), style_a_seq.get(r["row_id"], ""))
            a_jac_s = apply_rank_style(fmt3(r["a_jac"]), style_a_jac.get(r["row_id"], ""))
            a_emb_s = apply_rank_style(fmt3(r["a_emb"]), style_a_emb.get(r["row_id"], ""))

            b_seq_s = apply_rank_style(fmt3(r["b_seq"]), style_b_seq.get(r["row_id"], ""))
            b_jac_s = apply_rank_style(fmt3(r["b_jac"]), style_b_jac.get(r["row_id"], ""))
            b_emb_s = apply_rank_style(fmt3(r["b_emb"]), style_b_emb.get(r["row_id"], ""))

            lines.append(
                r"\hspace{0.5em}"
                + r["method_key"]
                + r" ("
                + co_name
                + r") & "
                + a_seq_s
                + " & "
                + a_jac_s
                + " & "
                + a_emb_s
                + " & "
                + b_seq_s
                + " & "
                + b_jac_s
                + " & "
                + b_emb_s
                + r" \\"
            )

        # 모델 블록 끝 굵은 라인
        lines.append(r"\Xhline{1.0pt}")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    out_dir = Path("runs_analysis") / "enternchment"
    summary_path = out_dir / "summary.json"
    tex_path = out_dir / "enternchment_summary.tex"

    summary_obj = load_json(summary_path)
    tex = build_table_tex(summary_obj)

    out_dir.mkdir(parents=True, exist_ok=True)
    with tex_path.open("w", encoding="utf-8") as f:
        f.write(tex)

    print(f"[OK] saved: {tex_path}")


if __name__ == "__main__":
    main()
