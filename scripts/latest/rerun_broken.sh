#!/usr/bin/env bash
# Re-run only the broken arms of the latest experiment after a fix.
#
# What it does:
#   1. Confirms the gaia2-cli dataset exists at a persistent path.
#   2. Drops the known-bad arms from each trace.jsonl so RESUME re-runs them:
#        - gaia2: B_evomem + C_gpr (dataset /tmp failure)   -> keep A only
#        - locomo: B_evomem (0 injection bug, now fixed)    -> keep A + C
#      gaia and terminal_bench_2 are left as-is (RESUME continues them).
#   3. Launches latest_runner.py with RESUME=1 + a persistent GAIA2_SCENARIO_DIR.
#
# Usage:
#   bash scripts/latest/rerun_broken.sh [MODEL]
#   MODEL defaults to hy3-preview-ioa.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

MODEL="${1:-hy3-preview-ioa}"
D="experiments_results/latest/$MODEL"
DATASET="/root/workspace/SkillForge/.datasets/gaia2-cli"

echo "==> rerun_broken: model=$MODEL dir=$D"

# 1) dataset sanity
if [ ! -f "$DATASET" ] && [ ! -d "$DATASET" ]; then
  echo "FATAL: persistent gaia2 dataset not found at $DATASET" >&2
  echo "       Run: mkdir -p \$(dirname $DATASET) && cp -r /tmp/harbor-datasets/datasets/gaia2-cli $DATASET" >&2
  exit 1
fi
SCENARIO_COUNT="$(find "$DATASET" -name scenario.json 2>/dev/null | wc -l)"
echo "==> gaia2 dataset: $SCENARIO_COUNT scenario.json present at $DATASET"

# 2) drop broken arms
/root/.conda/envs/skillforge/bin/python - <<PY
import json, os
D = "$D"
def clean(bench, drop_groups, keep_note):
    f = os.path.join(D, bench, "trace.jsonl")
    if not os.path.exists(f):
        print(f"  {bench:18s} (no trace yet)")
        return
    rows = [json.loads(l) for l in open(f) if l.strip()]
    kept = [r for r in rows if r.get("group") not in drop_groups]
    removed = len(rows) - len(kept)
    open(f, "w").write(("\n".join(json.dumps(r) for r in kept)) + ("\n" if kept else ""))
    print(f"  {bench:18s} removed {removed:4d} from {sorted(drop_groups)} | kept {len(kept)} ({keep_note})")
clean("gaia2", {"B_evomem", "C_gpr"}, "A_baseline only")
clean("locomo", {"B_evomem"}, "A + C")
print("  gaia / terminal_bench_2: untouched")
PY

# 3) relaunch
echo "==> launching resume run (PID will print below)"
GAIA2_SCENARIO_DIR="$DATASET" RESUME=1 ITER_CHAIN=3 CODEBUDDY_MODEL="$MODEL" \
  nohup /root/.conda/envs/skillforge/bin/python -u scripts/latest/latest_runner.py \
  > "$D/../run_hy3_full.log" 2>&1 &
echo "PID: $!"
echo "==> tail: tail -f $D/../run_hy3_full.log"
