#!/usr/bin/env bash
# Re-run only the broken arms of the latest experiment after a fix.
#
# What it does:
#   0. Refuses to run unless the code actually contains the LoCoMo-B injection fix
#      (else the rerun would reproduce the same bug — you forgot to `git pull`).
#   1. Confirms the gaia2-cli dataset exists at a PERSISTENT path AND actually
#      contains scenario.json files (the /tmp copy gets reaped mid-run).
#   2. Drops the known-bad arms from each trace.jsonl so RESUME re-runs them:
#        - gaia2 : B_evomem + C_gpr (dataset /tmp failure) -> keep A (A had 0
#                  errors and no memory, so it stays valid under the same data)
#        - locomo: B_evomem (0-injection bug, now fixed)    -> keep A + C
#      gaia and terminal_bench_2 are left as-is: RESUME continues them, and the
#      fix is a no-op on GAIA (its short-question path already injected).
#   3. Launches latest_runner.py with RESUME=1 and a persistent GAIA2_SCENARIO_DIR.
#
# Usage:
#   bash scripts/latest/rerun_broken.sh [MODEL]
#   MODEL defaults to hy3-preview-ioa.
# Env overrides:
#   GAIA2_SCENARIO_DIR  persistent gaia2-cli path (default: <repo>/.datasets/gaia2-cli)
#   PYTHON              python interpreter (default: skillforge conda env)
set -euo pipefail

cd "$(dirname "$0")/../.."          # repo root
REPO="$PWD"

MODEL="${1:-hy3-preview-ioa}"
D="experiments_results/latest/$MODEL"
DATASET="${GAIA2_SCENARIO_DIR:-$REPO/.datasets/gaia2-cli}"
PYTHON="${PYTHON:-/root/.conda/envs/skillforge/bin/python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON="$(command -v python3)"

echo "==> rerun_broken: model=$MODEL dir=$D"
echo "==> python=$PYTHON"

# 0) the whole point of the rerun is the fix — refuse if it isn't in the tree yet
if ! grep -q "_core_task(p.evidence)" scripts/latest/evomem_bridge.py; then
  echo "FATAL: LoCoMo-B injection fix not found in evomem_bridge.py." >&2
  echo "       Run 'git pull origin main' first — rerunning without it is pointless." >&2
  exit 1
fi
echo "==> code check: LoCoMo-B fix present"

# 1) dataset sanity — must exist AND contain scenarios, else gaia2 re-breaks 100%
if [ ! -d "$DATASET" ]; then
  echo "FATAL: persistent gaia2 dataset dir not found: $DATASET" >&2
  echo "       mkdir -p '$(dirname "$DATASET")' && cp -r /tmp/harbor-datasets/datasets/gaia2-cli '$DATASET'" >&2
  exit 1
fi
SCENARIO_COUNT="$(find "$DATASET" -name scenario.json 2>/dev/null | wc -l | tr -d ' ')"
if [ "${SCENARIO_COUNT:-0}" -eq 0 ]; then
  echo "FATAL: no scenario.json under $DATASET — dataset is empty/incomplete." >&2
  echo "       re-copy it, e.g.: cp -r /tmp/harbor-datasets/datasets/gaia2-cli '$DATASET'" >&2
  exit 1
fi
echo "==> gaia2 dataset OK: $SCENARIO_COUNT scenario.json at $DATASET"

# 2) drop broken arms so RESUME re-runs exactly them (RESUME skips completed
#    (group, task_id) pairs, so a bad arm must be removed to be redone)
"$PYTHON" - <<PY
import json, os
D = "$D"
def clean(bench, drop_groups, keep_note):
    f = os.path.join(D, bench, "trace.jsonl")
    if not os.path.exists(f):
        print(f"  {bench:18s} (no trace yet — RESUME will run it fresh)")
        return
    rows = [json.loads(l) for l in open(f) if l.strip()]
    kept = [r for r in rows if r.get("group") not in drop_groups]
    removed = len(rows) - len(kept)
    if kept:
        with open(f, "w") as fh:
            fh.write("\n".join(json.dumps(r) for r in kept) + "\n")
    else:
        os.remove(f)   # fully cleared -> remove so the bench runs truly fresh
    print(f"  {bench:18s} dropped {removed:4d} {sorted(drop_groups)} | kept {len(kept)} ({keep_note})")
clean("gaia2", {"B_evomem", "C_gpr"}, "A only")
clean("locomo", {"B_evomem"}, "A + C")
print("  gaia / terminal_bench_2: untouched (RESUME continues them)")
PY

# 3) relaunch (background, resumes everything not dropped)
LOG="$D/../run_${MODEL}_full.log"
echo "==> launching resume run -> $LOG"
GAIA2_SCENARIO_DIR="$DATASET" RESUME=1 ITER_CHAIN=3 CODEBUDDY_MODEL="$MODEL" \
  nohup "$PYTHON" -u scripts/latest/latest_runner.py > "$LOG" 2>&1 &
echo "PID: $!"
echo "==> tail -f $LOG"
