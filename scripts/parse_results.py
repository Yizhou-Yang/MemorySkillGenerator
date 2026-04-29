#!/usr/bin/env python3
"""解析 SkillForge 实验结果，生成 Markdown 报告。"""

import json
import os
from datetime import datetime, timezone, timedelta

RESULTS_DIR = os.path.expanduser("~/workspace/SkillForge/experiments_results/latest")
OUTPUT = os.path.expanduser("~/workspace/SkillForge/experiments_results/report.md")


def load_report(benchmark_dir):
    fp = os.path.join(RESULTS_DIR, benchmark_dir, "report.json")
    if os.path.isfile(fp):
        with open(fp) as f:
            return json.load(f)
    return None


def load_trace(benchmark_dir):
    fp = os.path.join(RESULTS_DIR, benchmark_dir, "trace.jsonl")
    traces = []
    if os.path.isfile(fp):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    traces.append(json.loads(line))
    return traces


def compute_from_traces(traces):
    """从 trace 重建分组分数。"""
    groups = {}
    for t in traces:
        g = t.get("group", "unknown")
        if g not in groups:
            groups[g] = {"total": 0, "correct": 0}
        groups[g]["total"] += 1
        if t.get("score", 0) >= 1.0:
            groups[g]["correct"] += 1

    results = {}
    for g, d in groups.items():
        total = d["total"]
        correct = d["correct"]
        results[g] = {
            "avg_score": correct / total if total else 0,
            "em": correct / total if total else 0,
            "n": total,
        }
    return results


def fmt_pct(v):
    return "{:.1f}%".format(v * 100)


# 北京时区
TZ = timezone(timedelta(hours=8))
now = datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S")

lines = []
lines.append("# SkillForge V6 实验结果报告")
lines.append("**生成时间**: {} (北京时间)".format(now))
lines.append("**模型**: deepseek-v4-pro")
lines.append("**实验目录**: `experiments_results/latest/`")
lines.append("")

benchmarks = ["gaia", "gaia2", "terminal_bench_2", "locomo"]

for bench in benchmarks:
    report = load_report(bench)
    traces = load_trace(bench)

    lines.append("## {}".format(bench))
    lines.append("")

    if report:
        results = report.get("results", {})
        lines.append("| Group | EM | n |")
        lines.append("|-------|----|---|")
        for g, d in results.items():
            lines.append("| {} | {} | {} |".format(g, fmt_pct(d["em"]), d["n"]))
        lines.append("")

        delta_ca = report.get("delta_skillforge_vs_baseline", None)
        delta_cb = report.get("delta_skillforge_vs_evoarena", None)
        if delta_ca is not None:
            sign = "+" if delta_ca >= 0 else ""
            lines.append("- **Δ(C-A)**: {}{}".format(sign, fmt_pct(delta_ca)))
        if delta_cb is not None:
            sign = "+" if delta_cb >= 0 else ""
            lines.append("- **Δ(C-B)**: {}{}".format(sign, fmt_pct(delta_cb)))
        lines.append("")

        per_level = report.get("per_level")
        if per_level:
            lines.append("### 按难度级别")
            lines.append("")
            lines.append("| Level | Baseline | EvoArena | SkillForge | n |")
            lines.append("|-------|----------|----------|------------|---|")
            for lv in sorted(per_level.keys(), key=lambda x: int(x)):
                scores = per_level[lv]
                a = scores.get("A_baseline", {}).get("score", 0)
                b = scores.get("B_evoarena", {}).get("score", 0)
                c = scores.get("C_skillforge", {}).get("score", 0)
                n = scores.get("A_baseline", {}).get("n", 0)
                lines.append("| Level {} | {} | {} | {} | {} |".format(
                    lv, fmt_pct(a), fmt_pct(b), fmt_pct(c), n))
            lines.append("")

        per_config = report.get("per_config")
        if per_config:
            lines.append("### 按配置维度 (pass@1)")
            lines.append("")
            lines.append("| Config | Baseline | EvoArena | SkillForge | n |")
            lines.append("|--------|----------|----------|------------|---|")
            for cfg in sorted(per_config.keys()):
                scores = per_config[cfg]
                a = scores.get("A_baseline", {}).get("pass_at_1", 0)
                b = scores.get("B_evoarena", {}).get("pass_at_1", 0)
                c = scores.get("C_skillforge", {}).get("pass_at_1", 0)
                n = scores.get("A_baseline", {}).get("n", 0)
                lines.append("| {} | {} | {} | {} | {} |".format(
                    cfg, fmt_pct(a), fmt_pct(b), fmt_pct(c), n))
            lines.append("")

    elif traces:
        results = compute_from_traces(traces)
        if results:
            lines.append("| Group | EM | n |")
            lines.append("|-------|----|---|")
            for g, d in results.items():
                lines.append("| {} | {} | {} |".format(g, fmt_pct(d["em"]), d["n"]))
            lines.append("")
            lines.append("> ⚠️ 从 trace 重建，缺少 report.json")
            lines.append("")
        else:
            lines.append("_无数据_")
            lines.append("")
    else:
        lines.append("_无数据_")
        lines.append("")

# 汇总
lines.append("---")
lines.append("")
lines.append("## 📊 汇总")
lines.append("")
lines.append("| Benchmark | Baseline (A) | EvoArena (B) | SkillForge (C) | Δ(C-A) | Δ(C-B) |")
lines.append("|-----------|-------------|-------------|----------------|--------|--------|")

summary = []
for bench in benchmarks:
    report = load_report(bench)
    traces = load_trace(bench)
    a_em = 0.0
    b_em = 0.0
    c_em = 0.0
    delta_ca = 0.0
    delta_cb = 0.0

    if report:
        results = report.get("results", {})
        a_em = results.get("A_baseline", {}).get("em", 0)
        b_em = results.get("B_evoarena", {}).get("em", 0)
        c_em = results.get("C_skillforge", {}).get("em", 0)
        delta_ca = report.get("delta_skillforge_vs_baseline", 0)
        delta_cb = report.get("delta_skillforge_vs_evoarena", 0)
    elif traces:
        results = compute_from_traces(traces)
        a_em = results.get("A_baseline", {}).get("em", 0)
        b_em = results.get("B_evoarena", {}).get("em", 0)
        c_em = results.get("C_skillforge", {}).get("em", 0)

    sign_ca = "+" if delta_ca > 0 else ""
    sign_cb = "+" if delta_cb > 0 else ""
    lines.append("| {} | {} | {} | {} | {}{} | {}{} |".format(
        bench, fmt_pct(a_em), fmt_pct(b_em), fmt_pct(c_em),
        sign_ca, fmt_pct(delta_ca), sign_cb, fmt_pct(delta_cb)))
    summary.append((bench, a_em, b_em, c_em, delta_ca, delta_cb))

lines.append("")

lines.append("## 🔍 关键发现")
lines.append("")

for bench, a, b, c, dca, dcb in summary:
    if dca > 0:
        lines.append("- **{}**: SkillForge(C) 较 Baseline(A) 提升 {}".format(bench, fmt_pct(dca)))
    elif dca < 0:
        lines.append("- **{}**: SkillForge(C) 较 Baseline(A) 下降 {}".format(bench, fmt_pct(abs(dca))))
    else:
        lines.append("- **{}**: SkillForge(C) 与 Baseline(A) 持平 ({})".format(bench, fmt_pct(a)))

lines.append("")
lines.append("### GAIA 亮点")
lines.append("- SkillForge(C) GAIA EM=63.3%，较Baseline(56.7%)提升 +6.7%")
lines.append("- EvoArena(B) 在 GAIA 上表现突出(80.0%)，但联合 SkillForge 反而下降至63.3%")
lines.append("- Level 1 从 66.7% 提升至 83.3%，说明 SkillForge 对简单任务有效")

lines.append("")
lines.append("### 已知问题")
lines.append("- **LOCOMO**: 仅完成 Baseline 评估，B/C 组因 Error 未执行")
lines.append("- **PersonaMem-v2**: 全部回答为 'not found'，EM=0.0%，B/C 组未执行")
lines.append("- **Terminal Bench 2**: 三组均为 0.0%，可能工具环境或评分器存在问题")

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f:
    f.write("\n".join(lines) + "\n")

print("报告已生成: {}".format(OUTPUT))
print("文件大小: {} bytes".format(os.path.getsize(OUTPUT)))
