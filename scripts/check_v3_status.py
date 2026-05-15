#!/usr/bin/env python3
"""Quick status check for paper v3 experiments (9 benchmarks)."""
import json, sys, os
from pathlib import Path

results_path = Path("experiments/paper_v3_results.json")
log_path = Path("experiments/paper_v3_stdout.log")

pid_check = os.popen("ps aux | grep run_paper_v3 | grep -v grep | wc -l").read().strip()
running = int(pid_check) > 0

print(f"{'='*70}")
print(f"Paper v3 Experiment Status (9 Benchmarks)")
print(f"{'='*70}")
print(f"Process running: {'YES' if running else 'FINISHED'}")

if log_path.exists():
    log = log_path.read_text()
    # Count completed benchmarks in main experiment
    bench_done = log.count("Dedup:")
    compacts = log.count("COMPACT")
    errors = log.count("FAILED")
    last_lines = log.strip().split("\n")[-5:]
    print(f"Benchmarks completed (main): {bench_done}/9")
    print(f"Compactions: {compacts}")
    print(f"Errors: {errors}")
    print(f"Last activity:")
    for l in last_lines:
        print(f"  {l}")

if results_path.exists():
    data = json.loads(results_path.read_text())
    meta = data.get("meta", {})
    print(f"\nTokens used: {meta.get('total_tokens', 0):,}")
    print(f"API calls: {meta.get('total_api_calls', 0)}")
    elapsed = meta.get('elapsed_seconds', 0)
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min / {elapsed/3600:.1f}h)")

    print(f"\nExperiment status:")
    for key in ["main_experiment", "attention_independence", "phenomena", "bound_tightening"]:
        if key in data:
            if isinstance(data[key], dict) and "error" in data[key]:
                print(f"  {key}: ERROR - {str(data[key]['error'])[:80]}")
            else:
                print(f"  {key}: DONE")
        else:
            print(f"  {key}: NOT STARTED")

    # Show Table 1 if available
    if "main_experiment" in data and isinstance(data["main_experiment"], dict):
        me = data["main_experiment"]
        benchmarks_with_results = [b for b in me if isinstance(me[b], dict) and "methods" in me[b]]
        if benchmarks_with_results:
            print(f"\n{'='*90}")
            print(f"TABLE 1 PREVIEW ({len(benchmarks_with_results)}/9 benchmarks)")
            print(f"{'='*90}")
            print(f"{'Benchmark':<18} {'Tier':>4} {'B0':>7} {'B1':>7} {'B2':>7} {'A1':>7} {'A2':>7} {'A3':>7} {'Dedup':>8}")
            print("-" * 90)
            methods = ["B0", "B1", "B2", "A1", "A2", "A3"]
            for bn in benchmarks_with_results:
                bd = me[bn]
                tier = bd.get("tier", "?")
                row = f"{bn:<18} T{tier:>3}"
                for m in methods:
                    if m in bd["methods"]:
                        row += f" {bd['methods'][m]['avg_em']:>6.1%}"
                    else:
                        row += f" {'N/A':>6}"
                ds = bd.get("dedup_stats", {})
                row += f" {ds.get('reduction_pct', 0):>7.1%}"
                print(row)

            # Errors
            errors_list = [b for b in me if isinstance(me[b], dict) and "error" in me[b]]
            if errors_list:
                print(f"\nFailed benchmarks: {errors_list}")

    # Show attention independence if available
    if "attention_independence" in data and isinstance(data["attention_independence"], dict):
        ai = data["attention_independence"]
        for bn, bd in ai.items():
            if isinstance(bd, dict) and "strategies" in bd:
                print(f"\nδ_attention [{bn}]: SR range={bd.get('sr_range',0):.1%}, "
                      f"verified={bd.get('independence_verified', False)}")

    # Show phenomena if available
    if "phenomena" in data and isinstance(data["phenomena"], dict):
        ph = data["phenomena"]
        if "phase_transition" in ph:
            for bn, pt in ph["phase_transition"].items():
                if isinstance(pt, dict) and "peak_info" in pt:
                    for sn, info in pt["peak_info"].items():
                        print(f"Phase Transition [{bn}/{sn}]: peak N={info['peak_size']}, EM={info['peak_em']:.1%}")
        if "scissors_effect" in ph:
            for bn, se in ph["scissors_effect"].items():
                if isinstance(se, dict):
                    for lib in ["append_only", "skillos", "ours"]:
                        if lib in se:
                            d = se[lib]
                            print(f"Scissors [{bn}/{lib}]: |S|={d.get('final_total',0)}, ratio={d.get('final_ratio',0):.3f}")
else:
    print("\nNo results file yet")

print(f"\n{'='*70}")
