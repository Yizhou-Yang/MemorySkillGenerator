#!/usr/bin/env bash
# ALFWorld induction → eval pipeline runner.
# Waits for induction to finish, then auto-starts eval with the produced banks.
# Can be re-attached: `tail -f experiments/alfworld_pipeline.out`

set -uo pipefail
cd /root/workspace/SkillForge
source .venv_alfworld/bin/activate

INDUCE_OUT="experiments/alfworld_induce.out"
EVAL_OUT="experiments/alfworld_eval.out"
B2="experiments/alfworld_skills/b2_bank.json"
A3="experiments/alfworld_skills/a3_bank.json"
CALIB="experiments/alfworld_skills/train_calib.json"
EVAL_RESULT="experiments/alfworld_eval_results.json"

echo "[$(date)] Pipeline started." | tee -a experiments/alfworld_pipeline.out

# ----- Wait for induction to finish (look for FINAL marker or process exit) -----
INDUCE_PID="$(pgrep -f 'induce_alfworld_skills' | head -1 || true)"
if [ -z "$INDUCE_PID" ]; then
    echo "[$(date)] ERROR: induction process not running. Aborting." | tee -a experiments/alfworld_pipeline.out
    exit 1
fi
echo "[$(date)] Watching induction PID=$INDUCE_PID" | tee -a experiments/alfworld_pipeline.out

# Poll every 60s, log progress
while kill -0 "$INDUCE_PID" 2>/dev/null; do
    sleep 60
    last=$(tail -3 "$INDUCE_OUT" | tr '\n' ' ' | cut -c1-200)
    echo "[$(date)] [induce running] $last" | tee -a experiments/alfworld_pipeline.out
done

echo "[$(date)] Induction finished." | tee -a experiments/alfworld_pipeline.out

# Verify outputs exist
for f in "$B2" "$A3" "$CALIB"; do
    if [ ! -f "$f" ]; then
        echo "[$(date)] ERROR: induction did not produce $f. Aborting." | tee -a experiments/alfworld_pipeline.out
        tail -50 "$INDUCE_OUT" | tee -a experiments/alfworld_pipeline.out
        exit 1
    fi
done
echo "[$(date)] All induction outputs present." | tee -a experiments/alfworld_pipeline.out

# ----- Launch eval -----
echo "[$(date)] Launching eval..." | tee -a experiments/alfworld_pipeline.out
python -u scripts/run_alfworld_eval.py \
    --split valid_unseen \
    --n-test 50 \
    --max-steps 50 \
    --seed 42 \
    --top-k 3 \
    --methods B0 B2 A3 A3+PlanC \
    --skill-bank-b2 "$B2" \
    --skill-bank-a3 "$A3" \
    --train-calib "$CALIB" \
    --output "$EVAL_RESULT" \
    > "$EVAL_OUT" 2>&1

EVAL_RC=$?
echo "[$(date)] Eval finished with rc=$EVAL_RC" | tee -a experiments/alfworld_pipeline.out
if [ $EVAL_RC -eq 0 ] && [ -f "$EVAL_RESULT" ]; then
    echo "[$(date)] SUCCESS. Final results: $EVAL_RESULT" | tee -a experiments/alfworld_pipeline.out
    python -c "
import json
r = json.load(open('$EVAL_RESULT'))
print('=== Summary ===')
for m, d in r.get('methods', {}).items():
    print(f'  {m:12s}: SR={d[\"sr\"]:.1%}  steps={d[\"avg_steps\"]:.1f}  '
          f'tok={d[\"avg_tokens\"]:.0f}  ({d[\"n_won\"]}/{d[\"n_total\"]})')
" 2>&1 | tee -a experiments/alfworld_pipeline.out
else
    echo "[$(date)] FAILURE. tail of eval log:" | tee -a experiments/alfworld_pipeline.out
    tail -50 "$EVAL_OUT" | tee -a experiments/alfworld_pipeline.out
fi
