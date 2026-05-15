#!/usr/bin/env python3
"""Quick status check for paper v2 experiments."""
import json, sys, os
from pathlib import Path

results_path = Path("experiments/paper_v2_results.json")
log_path = Path("experiments/paper_v2_stdout.log")

# Check if process is running
pid_check = os.popen("ps aux | grep run_paper_v2 | grep -v grep | wc -l").read().strip()
running = int(pid_check) > 0

print(f"{'='*60}")
print(f"Paper v2 Experiment Status")
print(f"{'='*60}")
print(f"Process running: {'YES' if running else 'FINISHED'}")

# Check log progress
if log_path.exists():
    log = log_path.read_text()
    skills_done = log.count("Skill induction complete")
    compacts = log.count("COMPACT")
    errors = log.count("FAILED")
    last_lines = log.strip().split("\n")[-3:]
    print(f"Skills induced: {skills_done}")
    print(f"Compactions: {compacts}")
    print(f"Errors: {errors}")
    print(f"Last activity:")
    for l in last_lines:
        print(f"  {l}")

# Check results
if results_path.exists():
    data = json.loads(results_path.read_text())
    meta = data.get("meta", {})
    print(f"\nTokens used: {meta.get('total_tokens', 0):,}")
    print(f"API calls: {meta.get('total_api_calls', 0)}")
    print(f"Elapsed: {meta.get('elapsed_seconds', 0):.0f}s ({meta.get('elapsed_seconds', 0)/60:.1f}min)")

    # Show completed experiments
    print(f"\nCompleted experiments:")
    for key in ["main_experiment", "attention_independence", "phenomena", "bound_tightening", "cross_benchmark"]:
        if key in data:
            if "error" in data[key]:
                print(f"  {key}: ERROR - {data[key]['error'][:80]}")
            else:
                print(f"  {key}: OK")
        else:
            print(f"  {key}: NOT STARTED")

    # Show key results if available
    if "main_experiment" in data and "error" not in data["main_experiment"]:
        me = data["main_experiment"]
        for bench, bd in me.items():
            if isinstance(bd, dict) and "methods" in bd:
                print(f"\n  Table 1 [{bench}]:")
                for method, md in bd["methods"].items():
                    print(f"    {method}: EM={md['avg_em']:.1%}±{md.get('std_em',0):.2f}, "
                          f"F1={md['avg_f1']:.3f}, tokens={md['avg_tokens']:.0f}")
                if "dedup_stats" in bd:
                    ds = bd["dedup_stats"]
                    print(f"    Dedup: {ds['original_size']}→{ds['deduped_size']} ({ds['reduction_pct']:.1%} reduction)")

    if "attention_independence" in data and "error" not in data["attention_independence"]:
        ai = data["attention_independence"]
        print(f"\n  Table 2: SR range={ai.get('sr_range',0):.1%}, "
              f"F1 range={ai.get('f1_range',0):.3f}, "
              f"verified={ai.get('independence_verified', False)}")

    if "phenomena" in data and "error" not in data["phenomena"]:
        ph = data["phenomena"]
        if "compaction_cliff" in ph:
            cc = ph["compaction_cliff"]
            print(f"\n  Compaction Cliff: avg ratio={cc.get('avg_cliff_ratio',1):.2f}, "
                  f"{len(cc.get('compaction_points',[]))} compactions")
        if "scissors_effect" in ph:
            se = ph["scissors_effect"]
            for lib in ["append_only", "skillos", "ours"]:
                if lib in se:
                    d = se[lib]
                    print(f"  Scissors [{lib}]: |S|={d.get('final_total',0)}, "
                          f"N_eff={d.get('final_effective',0)}, ratio={d.get('final_ratio',0):.3f}")
else:
    print("\nNo results file yet (experiment still in early phase)")

print(f"\n{'='*60}")
