#!/usr/bin/env python3
"""SkillForge V6 Rerun — DeepSeek V4 Pro via CodeBuddy CLI"""
import asyncio
import json
import os
import sys
import time
import copy
from pathlib import Path

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# 环境变量 — 使用 codebuddy CLI 调用 deepseek-v4-pro
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

os.environ['LLM_PROVIDER'] = 'codebuddy'
os.environ['CODEBUDDY_MODEL'] = 'deepseek-v4-pro'
os.environ.setdefault('CODEBUDDY_INTERNET_ENVIRONMENT', 'ioa')

from codebuddy_agent_sdk import query, CodeBuddyAgentOptions, AssistantMessage

from v6 import (SkillForgeV6, ExperienceLibrary, Experience,
                analyze_execution, build_augmented_prompt,
                format_success_experience, format_failure_experience,
                ai_review_experience)
from benchmarks.loader import BenchmarkLoader

# ─── Config ───────────────────────────────────────────────────────────────
MODEL = "deepseek-v4-pro"
CONCURRENCY = 20  # deepseek-v4-pro 支持 500 并发，但保守一些
TASK_TIMEOUT = 180  # 3 min per task（QA 类任务不需要太长）
RESULTS_DIR = str(PROJECT_ROOT / "experiments_results" / "rerun_deepseek_v4pro")

# 每个 benchmark 的任务数
TASK_LIMITS = {
    "gaia": 50,       # 25 train + 25 test
    "alfworld": 40,   # 20 train + 20 test
    "locomo": 50,     # 25 train + 25 test
}

# 强制重新 train（旧经验库全是 failure，质量太差，需要用 deepseek-v4-pro 重新收集）
# 不再使用旧的经验库
FORCE_RETRAIN = True

# ─── LLM Helper (for AI Review) ──────────────────────────────────────────

def llm_review_fn(prompt: str) -> str:
    """调用 deepseek-v4-pro 进行 AI review（单轮，同步）。"""
    import concurrent.futures

    async def _call():
        opt = CodeBuddyAgentOptions(
            permission_mode="bypassPermissions", model=MODEL, max_turns=2, cwd="/tmp"
        )
        result = ""
        try:
            async with asyncio.timeout(90):
                async for msg in query(prompt=prompt, options=opt):
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if hasattr(block, 'text') and block.text:
                                result += block.text
                        if result:
                            break
        except Exception:
            pass
        return result

    def _run_in_thread():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_call())
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_thread)
        return future.result(timeout=120)

# ─── Task Runner ──────────────────────────────────────────────────────────

async def run_task(task: dict, experience_section: str = "",
                   benchmark: str = "gaia", group: str = "A") -> dict:
    """运行单个任务（QA 类：GAIA HF / ALFWorld / LoCoMo）。"""
    task_id = task["task_id"]
    description = task["description"]
    expected = task.get("expected", "")

    system = "You are a helpful assistant. Answer the question directly and concisely."
    if experience_section:
        system += f"\n\n{experience_section}"

    prompt = f"[System]\n{system}\n\n{description}"

    opt = CodeBuddyAgentOptions(
        permission_mode="bypassPermissions", model=MODEL, max_turns=2, cwd="/tmp"
    )

    result = {"task_id": task_id, "expected": expected, "response": "", "error": None,
              "time_cost": 0, "augmented": bool(experience_section), "group": group}
    t0 = time.time()

    try:
        async with asyncio.timeout(TASK_TIMEOUT):
            async for msg in query(prompt=prompt, options=opt):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if hasattr(block, 'text') and block.text:
                            if '429' in block.text and '额度' in block.text:
                                result["error"] = "429_rate_limit"
                                break
                            result["response"] += block.text
                    if result["response"] or result["error"]:
                        break
    except TimeoutError:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)[:200]

    result["time_cost"] = time.time() - t0
    return result

# ─── Evaluation ───────────────────────────────────────────────────────────

def evaluate_task(result: dict, benchmark: str) -> dict:
    """计算单个任务的分数。"""
    expected = result.get("expected", "").strip().lower()
    response = result.get("response", "").strip().lower()

    if not expected or not response:
        return {"score": 0.0, "method": "empty"}

    # Exact match (substring)
    em = 1.0 if expected in response or response in expected else 0.0

    # Token F1
    exp_tokens = set(expected.split())
    resp_tokens = set(response.split())
    if exp_tokens and resp_tokens:
        precision = len(exp_tokens & resp_tokens) / len(resp_tokens)
        recall = len(exp_tokens & resp_tokens) / len(exp_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        f1 = 0.0

    return {"score": max(em, f1), "em": em, "f1": f1, "method": "token_f1"}

# ─── Benchmark Runner ─────────────────────────────────────────────────────

async def run_benchmark(benchmark: str, tasks: list[dict],
                        existing_library_path: str | None = None,
                        run_static_dynamic: bool = False) -> dict:
    """运行单个 benchmark 的完整 ablation。"""
    print(f"\n{'='*70}")
    print(f"  Benchmark: {benchmark} (model: {MODEL})")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Train: {len(tasks)//2} | Test: {len(tasks) - len(tasks)//2}")
    if run_static_dynamic:
        print(f"  Mode: 静态路径 + 动态路径 对比")
    print(f"{'='*70}")

    mid = len(tasks) // 2
    train_tasks = tasks[:mid]
    test_tasks = tasks[mid:]

    os.makedirs(f"{RESULTS_DIR}/{benchmark}", exist_ok=True)

    # ─── Phase 1: 用 deepseek-v4-pro 重新 train（收集高质量经验）────
    sf = SkillForgeV6(token_budget=2000)
    sem = asyncio.Semaphore(CONCURRENCY)

    print(f"\n  Phase 1: 用 {MODEL} 运行 {len(train_tasks)} 个 train 任务（收集高质量经验）...")

    async def run_train(task):
        async with sem:
            r = await run_task(task, experience_section="", benchmark=benchmark, group="train")
            # 用 evaluate_task 评估 train 结果，正确标记 success/failure
            eval_result = evaluate_task(r, benchmark)
            score = eval_result.get("score", 0.0)
            is_success = score > 0.3  # 阈值：F1 > 0.3 视为成功

            response = r.get("response", "")
            expected = task.get("expected", "")

            # 直接构造 Experience 对象（绕过 analyze_execution 的 action matching，
            # 因为 QA 类任务没有 tool actions，action matching 会全部返回 score=0）
            from v6.experience import Experience
            outcome = "success" if score >= 0.8 else "partial" if score >= 0.3 else "failure"
            exp = Experience(
                task_id=r["task_id"],
                task_desc=task["description"][:300],
                tool_sequence=["answer"],
                action_commands=[response[:300]],
                outcome=outcome,
                score=score,
                missing_steps=[] if is_success else ["correct_answer"],
                extra_steps=[],
                failure_reason="" if is_success else f"Answer mismatch (F1={score:.2f})",
                failure_taxonomy={
                    "category": "success" if is_success else "model_failure",
                    "root_cause": "" if is_success else f"F1={score:.2f}, expected='{expected[:50]}'"
                },
                token_cost=len(response) // 4,
                time_cost=r.get("time_cost", 0),
                task_complexity="simple",
                augmentation_used="",
                timestamp=time.time(),
            )

            # AI refinement
            review = ai_review_experience(exp, llm_fn=llm_review_fn)
            exp.failure_taxonomy.update({
                "ai_refined": review.get("refined", False),
                "causal_lesson": review.get("causal_lesson", ""),
                "avoidance_note": review.get("avoidance_note", ""),
                "transferability": review.get("transferability", ""),
                "generalized_steps": review.get("generalized_steps", ""),
                "evolution_insight": review.get("evolution_insight", ""),
                "quality_score": review.get("quality_score", 0),
            })

            sf.library.record(exp)
            r["_train_score"] = score
            r["_train_success"] = is_success
            return r

    train_results = await asyncio.gather(*[run_train(t) for t in train_tasks])
    train_valid = [r for r in train_results if not r.get("error")]
    train_success = [r for r in train_results if r.get("_train_success")]
    print(f"  Train 完成: {len(train_valid)}/{len(train_tasks)} 有效响应")
    print(f"  Train 成功率: {len(train_success)}/{len(train_valid)} (score > 0.3)")
    avg_train_score = sum(r.get("_train_score", 0) for r in train_results) / max(len(train_results), 1)
    print(f"  Train 平均分: {avg_train_score:.1%}")
    print(f"  经验库: {sf.stats}")

    # 保存经验库
    sf.save(f"{RESULTS_DIR}/{benchmark}/library_after_train.json")

    # ─── Phase 2: Test (3 groups: A/B/C) ─────────────────────────
    print(f"\n  Phase 2: 测试 {len(test_tasks)} 个任务 × 3 组...")

    # Group A: Baseline（无增强）
    print(f"    [A] Baseline (no augmentation)...")
    async def run_test_baseline(task):
        async with sem:
            return await run_task(task, experience_section="", benchmark=benchmark, group="A")
    results_a = await asyncio.gather(*[run_test_baseline(t) for t in test_tasks])

    # Group B: Raw injection（原始经验注入，无 AI 精炼）
    print(f"    [B] Raw experience injection...")
    raw_library = ExperienceLibrary()
    for exp in sf.library.experiences:
        raw_exp = copy.deepcopy(exp)
        # 清除 AI-refined 字段
        raw_exp.failure_taxonomy = {k: v for k, v in raw_exp.failure_taxonomy.items()
                                     if k not in ("ai_refined", "causal_lesson", "avoidance_note",
                                                  "transferability", "generalized_steps",
                                                  "evolution_insight", "quality_score", "evolution_trace")}
        raw_library.record(raw_exp)

    async def run_test_raw(task):
        async with sem:
            aug = build_augmented_prompt(task["description"][:300], raw_library, token_budget=2000)
            return await run_task(task, experience_section=aug, benchmark=benchmark, group="B")
    results_b = await asyncio.gather(*[run_test_raw(t) for t in test_tasks])

    # Group C: AI-refined injection（完整 AI 精炼经验注入）
    print(f"    [C] AI-refined experience injection...")
    async def run_test_refined(task):
        async with sem:
            aug = build_augmented_prompt(task["description"][:300], sf.library, token_budget=2000)
            return await run_task(task, experience_section=aug, benchmark=benchmark, group="C")
    results_c = await asyncio.gather(*[run_test_refined(t) for t in test_tasks])

    # ─── Evaluate ─────────────────────────────────────────────────
    scores = {"A_baseline": [], "B_raw": [], "C_refined": []}
    for i, task in enumerate(test_tasks):
        scores["A_baseline"].append(evaluate_task(results_a[i], benchmark))
        scores["B_raw"].append(evaluate_task(results_b[i], benchmark))
        scores["C_refined"].append(evaluate_task(results_c[i], benchmark))

    report = {}
    for group, evals in scores.items():
        valid = [e["score"] for e in evals if e.get("score") is not None]
        avg = sum(valid) / len(valid) if valid else 0
        report[group] = {"avg_score": avg, "n": len(valid)}

    print(f"\n  结果 ({benchmark}, model={MODEL}):")
    print(f"    A (Baseline):    {report['A_baseline']['avg_score']:.1%}")
    print(f"    B (Raw inject):  {report['B_raw']['avg_score']:.1%}")
    print(f"    C (AI-refined):  {report['C_refined']['avg_score']:.1%}")
    delta_ac = report['C_refined']['avg_score'] - report['A_baseline']['avg_score']
    delta_bc = report['C_refined']['avg_score'] - report['B_raw']['avg_score']
    print(f"    Δ(C-A): {delta_ac:+.1%} | Δ(C-B): {delta_bc:+.1%}")

    full_report = {
        "benchmark": benchmark,
        "model": MODEL,
        "n_train": len(train_tasks), "n_test": len(test_tasks),
        "results": report,
        "delta_refined_vs_baseline": delta_ac,
        "delta_refined_vs_raw": delta_bc,
    }

    # ─── LoCoMo 静态/动态路径对比 ─────────────────────────────────
    if run_static_dynamic:
        print(f"\n  === LoCoMo 静态 vs 动态路径对比 ===")
        print(f"    静态路径 = Group A (直接 QA，无经验注入)")
        print(f"    动态路径 = Group C (AI-Refined experience injection)")
        static_score = report['A_baseline']['avg_score']
        dynamic_score = report['C_refined']['avg_score']
        delta_sd = dynamic_score - static_score
        print(f"    静态路径 (A): {static_score:.1%}")
        print(f"    动态路径 (C): {dynamic_score:.1%}")
        print(f"    Δ(动态-静态): {delta_sd:+.1%}")
        full_report["static_vs_dynamic"] = {
            "static_score": static_score,
            "dynamic_score": dynamic_score,
            "delta": delta_sd,
            "note": "静态=baseline QA, 动态=AI-Refined injection"
        }

    # 保存结果
    with open(f"{RESULTS_DIR}/{benchmark}/report.json", "w") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    # 保存详细结果（每个任务的 response）
    detail = {
        "A": [{"task_id": r["task_id"], "response": r["response"][:500],
               "error": r.get("error"), "time_cost": r["time_cost"]} for r in results_a],
        "B": [{"task_id": r["task_id"], "response": r["response"][:500],
               "error": r.get("error"), "time_cost": r["time_cost"]} for r in results_b],
        "C": [{"task_id": r["task_id"], "response": r["response"][:500],
               "error": r.get("error"), "time_cost": r["time_cost"]} for r in results_c],
    }
    with open(f"{RESULTS_DIR}/{benchmark}/detail.json", "w") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)

    return full_report

# ─── Main ─────────────────────────────────────────────────────────────────

async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║  SkillForge V6 Rerun — DeepSeek V4 Pro via CodeBuddy CLI     ║")
    print("║  Benchmarks: GAIA HF, ALFWorld, LoCoMo                       ║")
    print(f"║  Model: {MODEL:<20} | Concurrency: {CONCURRENCY:<5}          ║")
    print("║  LoCoMo: 静态路径 + 动态路径 对比                            ║")
    print("╚════════════════════════════════════════════════════════════════╝")

    all_reports = {}

    # 加载 benchmarks
    print("\n  加载 benchmarks...")
    benchmarks = {}

    for name in ["gaia", "alfworld", "locomo"]:
        loader = BenchmarkLoader({"name": name, "num_samples": TASK_LIMITS[name]})
        tasks = loader.load()[:TASK_LIMITS[name]]
        benchmarks[name] = tasks
        print(f"    {name}: {len(tasks)} tasks")

    print(f"\n  总计: {sum(len(t) for t in benchmarks.values())} tasks")

    # 运行每个 benchmark（强制重新 train，不使用旧经验库）
    for name, tasks in benchmarks.items():
        if not tasks:
            print(f"\n  SKIP {name}: 无任务")
            continue
        try:
            report = await run_benchmark(
                name, tasks,
                existing_library_path=None,  # 强制重新 train
                run_static_dynamic=(name == "locomo"),  # LoCoMo 同时跑静态/动态对比
            )
            all_reports[name] = report
        except Exception as e:
            import traceback
            print(f"\n  ERROR on {name}: {e}")
            traceback.print_exc()
            all_reports[name] = {"error": str(e)}

    # 最终汇总
    print(f"\n\n{'═'*70}")
    print(f"  FINAL SUMMARY (DeepSeek V4 Pro)")
    print(f"{'═'*70}")
    print(f"  {'Benchmark':<12} {'Baseline':>10} {'Raw':>10} {'AI-Refined':>12} {'Δ(C-A)':>8} {'Δ(C-B)':>8}")
    print(f"  {'-'*60}")
    for name, r in all_reports.items():
        if "error" in r:
            print(f"  {name:<12} ERROR: {r['error'][:40]}")
        else:
            res = r["results"]
            print(f"  {name:<12} {res['A_baseline']['avg_score']:>9.1%} "
                  f"{res['B_raw']['avg_score']:>9.1%} "
                  f"{res['C_refined']['avg_score']:>11.1%} "
                  f"{r['delta_refined_vs_baseline']:>+7.1%} "
                  f"{r['delta_refined_vs_raw']:>+7.1%}")

    # LoCoMo 静态/动态对比
    if "locomo" in all_reports and "static_vs_dynamic" in all_reports.get("locomo", {}):
        sd = all_reports["locomo"]["static_vs_dynamic"]
        print(f"\n  LoCoMo 静态 vs 动态:")
        print(f"    静态路径 (直接QA):         {sd['static_score']:.1%}")
        print(f"    动态路径 (AI-Refined):     {sd['dynamic_score']:.1%}")
        print(f"    Δ(动态-静态):              {sd['delta']:+.1%}")

    # 保存最终汇总
    with open(f"{RESULTS_DIR}/final_summary.json", "w") as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\n  已保存: {RESULTS_DIR}/final_summary.json")

    # 与之前 hy3-preview-ioa 结果对比
    prev_results_dir = str(PROJECT_ROOT / "experiments_results" / "unified_v6_results")
    print(f"\n\n{'═'*70}")
    print(f"  模型对比: hy3-preview-ioa vs deepseek-v4-pro")
    print(f"{'═'*70}")
    print(f"  {'Benchmark':<12} {'hy3 (A)':>10} {'hy3 (C)':>10} {'ds4p (A)':>10} {'ds4p (C)':>10} {'Δ model':>8}")
    print(f"  {'-'*62}")
    for name in ["gaia", "alfworld", "locomo"]:
        prev_path = f"{prev_results_dir}/{name}/report.json"
        if os.path.exists(prev_path):
            with open(prev_path) as f:
                prev = json.load(f)
            prev_a = prev["results"]["A_baseline"]["avg_score"]
            prev_c = prev["results"]["C_refined"]["avg_score"]
        else:
            prev_a, prev_c = 0, 0

        if name in all_reports and "results" in all_reports[name]:
            new_a = all_reports[name]["results"]["A_baseline"]["avg_score"]
            new_c = all_reports[name]["results"]["C_refined"]["avg_score"]
            delta_model = new_c - prev_c
            print(f"  {name:<12} {prev_a:>9.1%} {prev_c:>9.1%} {new_a:>9.1%} {new_c:>9.1%} {delta_model:>+7.1%}")

if __name__ == "__main__":
    asyncio.run(main())
