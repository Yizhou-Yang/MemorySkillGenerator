#!/usr/bin/env bash
# One-command verdict for a model's 4-benchmark run: every gate + every analysis,
# in the order of experiments_results/EXPERIMENT_QUALITY.md. Run it on the server
# (or anywhere the trace.jsonl files are synced).
#
#   bash scripts/latest/check_run.sh <MODEL> [RESULTS_BASE]
#   MODEL defaults to deepseek-v4-pro; RESULTS_BASE defaults to latest_evolving.
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL="${1:-deepseek-v4-pro}"
BASE="${2:-latest_evolving}"
D="experiments_results/$BASE/$MODEL"
H="experiments_results/harbor_tb2/$MODEL"
PY="${PYTHON:-python3}"

echo "================================================================"
echo " check_run: model=$MODEL base=$BASE"
echo "================================================================"

echo ""
echo "── Gate 0-1: completeness / errors / protocol fields ──"
"$PY" - "$D" "$H" <<'EOF'
import json, sys, collections, glob, os
qa_dir, tb_dir = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f"{qa_dir}/*/trace.jsonl")) + sorted(glob.glob(f"{tb_dir}/*/trace.jsonl"))
if not files:
    print(f"  no trace.jsonl under {qa_dir} or {tb_dir}")
for f in files:
    bench = f.split("/")[-2]
    rows = [json.loads(l) for l in open(f) if l.strip()]
    by = collections.defaultdict(lambda: [0, 0])
    revs, fb = collections.Counter(), collections.Counter()
    mut = it1 = inj = 0
    for r in rows:
        g = r.get("group", "?"); by[g][0] += 1
        if r.get("error"): by[g][1] += 1
        revs[r.get("code_rev", "pre")] += 1
        fb[r.get("fb_mode", "-")] += 1
        if (r.get("iteration") or 0) >= 1:
            it1 += 1
            if r.get("mutated"): mut += 1
        if r.get("patch_injected"): inj += 1
    uniq = collections.Counter()
    for r in rows:
        uniq[(r.get("group","?"), r.get("task_id"), r.get("iteration"))] += 1
    uarm = collections.Counter(g for (g,_,_) in uniq)
    arms = " | ".join(f"{g.split('_')[0]}:n={n}(uniq={uarm.get(g,0)}),err={100*e//max(n,1)}%"
                      for g, (n, e) in sorted(by.items()))
    flags = []
    if any(e / max(n, 1) > 0.1 for n, e in by.values()): flags.append("HIGH-ERR")
    if len(revs) > 1: flags.append(f"MIXED-REV{dict(revs)}")
    print(f"  {bench:18s} {arms}")
    print(f"  {'':18s} rev={list(revs)[0] if len(revs)==1 else dict(revs)}"
          f" fb={dict(fb)} mutated={mut}/{it1} injected={inj}"
          + ("  <-- " + ",".join(flags) if flags else ""))
EOF

echo ""
echo "── TB2 (harbor): env-death visibility + arm fairness ──"
"$PY" - "$H" <<'EOF'
import json, sys, collections, glob
tb = glob.glob(f"{sys.argv[1]}/*/trace.jsonl")
if not tb:
    print("  (no harbor TB2 trace yet)")
else:
    rows = [json.loads(l) for l in open(tb[0]) if l.strip()]
    fm = collections.defaultdict(collections.Counter)
    tasks = collections.defaultdict(set)
    for r in rows:
        g = r.get("group", "?")
        fm[g][r.get("failure_mode") or "(none)"] += 1
        tasks[g].add(r.get("task_id"))
    for g in sorted(fm):
        top = ", ".join(f"{k}x{v}" for k, v in fm[g].most_common(4))
        print(f"  {g:12s} failure_mode: {top}")
    sets = list(tasks.values())
    if len(sets) > 1:
        common = set.intersection(*sets); union = set.union(*sets)
        if common != union:
            print(f"  <-- ARM TASK-SET MISMATCH: {len(union)-len(common)} tasks "
                  f"missing from some arm (harness died pre-result) — paired "
                  f"stats must restrict to the {len(common)} common tasks")
        else:
            print(f"  arm task sets identical ({len(common)} tasks) — paired-safe")
EOF

echo ""
echo "── Gate 2-3: injection isolation + paired significance ──"
"$PY" scripts/latest/breakdown.py "$D" 2>/dev/null | grep -E "^## |injected acc|NO patches|chain-level" | head -30
echo ""
"$PY" scripts/latest/soft_stats.py "$D" 2>/dev/null
[ -d "$H/terminal_bench_2" ] && "$PY" scripts/latest/soft_stats.py "$H" terminal_bench_2 2>/dev/null

echo ""
echo "── GAIA2 dynamic-vs-static splits (needs dataset dir) ──"
if [ -n "${GAIA2_SCENARIO_DIR:-}" ] || [ -d ".datasets/gaia2-cli" ]; then
  "$PY" scripts/latest/gaia2_split_report.py "$D" 2>/dev/null | tail -12
else
  echo "  (skipped: set GAIA2_SCENARIO_DIR to enable)"
fi

echo ""
echo "── verdict checklist ──"
cat <<'TXT'
  [ ] every benchmark has A/B/C at expected n, err≈0, ONE code_rev
  [ ] fb_mode=self everywhere (QA) / env-by-construction (TB2)
  [ ] mutated≈100% of iter>=1 rows; injected>0 for B and C
  [ ] C vs B soft CI: the paper claim lives or dies here
  [ ] gaia2: dynamic splits (adaptability/time/ambiguity) vs static controls
TXT
