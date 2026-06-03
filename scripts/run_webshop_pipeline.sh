#!/usr/bin/env bash
# WebShop induction → eval pipeline runner.
# Mirrors run_alfworld_pipeline.sh: waits for induction to finish, then
# auto-starts eval with the produced banks. Re-attach with:
#   tail -f experiments/webshop_pipeline.out

set -uo pipefail
cd /root/workspace/SkillForge

# WebShop deps live in the system python (datasets, sentence-transformers); no
# special venv required. Use the same interpreter as v5.
PY="${PY:-/usr/bin/python3.9}"

INDUCE_OUT="experiments/webshop_induce.out"
EVAL_OUT="experiments/webshop_eval.out"
B2="experiments/webshop_skills/b2_bank.json"
A3="experiments/webshop_skills/a3_bank.json"
CALIB="experiments/webshop_skills/train_calib.json"
EXCL="experiments/webshop_skills/induced_ids.json"
EVAL_RESULT="experiments/webshop_eval_results.json"

mkdir -p experiments/webshop_skills

echo "[$(date)] WebShop pipeline started." | tee -a experiments/webshop_pipeline.out

# ----- Step 0: launch induction if not already running -----
INDUCE_PID="$(pgrep -f 'induce_webshop_skills' | head -1 || true)"
if [ -z "$INDUCE_PID" ]; then
    echo "[$(date)] Launching induction..." | tee -a experiments/webshop_pipeline.out
    nohup "$PY" -u scripts/induce_webshop_skills.py \
        --n-train 30 \
        --min-traj-steps 3 \
        --seed 2024 \
        --output-b2 "$B2" \
        --output-a3 "$A3" \
        > "$INDUCE_OUT" 2>&1 &
    INDUCE_PID=$!
    echo "[$(date)] Induction PID=$INDUCE_PID" | tee -a experiments/webshop_pipeline.out
else
    echo "[$(date)] Induction already running (PID=$INDUCE_PID), watching." \
        | tee -a experiments/webshop_pipeline.out
fi

# ----- Step 1: wait for induction -----
while kill -0 "$INDUCE_PID" 2>/dev/null; do
    sleep 30
    last=$(tail -3 "$INDUCE_OUT" 2>/dev/null | tr '\n' ' ' | cut -c1-200)
    echo "[$(date)] [induce running] $last" | tee -a experiments/webshop_pipeline.out
done

echo "[$(date)] Induction finished." | tee -a experiments/webshop_pipeline.out

# Verify outputs
for f in "$B2" "$A3" "$CALIB" "$EXCL"; do
    if [ ! -f "$f" ]; then
        echo "[$(date)] ERROR: induction did not produce $f. Aborting." \
            | tee -a experiments/webshop_pipeline.out
        tail -50 "$INDUCE_OUT" | tee -a experiments/webshop_pipeline.out
        exit 1
    fi
done
echo "[$(date)] All induction outputs present." | tee -a experiments/webshop_pipeline.out

# ----- Step 2: launch eval -----
echo "[$(date)] Launching eval..." | tee -a experiments/webshop_pipeline.out
"$PY" -u scripts/run_webshop_eval.py \
    --split test \
    --n-test 100 \
    --seed 42 \
    --top-k 3 \
    --methods B0 B2 A3 A3+PlanC \
    --skill-bank-b2 "$B2" \
    --skill-bank-a3 "$A3" \
    --train-calib "$CALIB" \
    --exclude-ids-file "$EXCL" \
    --output "$EVAL_RESULT" \
    > "$EVAL_OUT" 2>&1

EVAL_RC=$?
echo "[$(date)] Eval finished with rc=$EVAL_RC" | tee -a experiments/webshop_pipeline.out
if [ $EVAL_RC -eq 0 ] && [ -f "$EVAL_RESULT" ]; then
    echo "[$(date)] SUCCESS. Final results: $EVAL_RESULT" \
        | tee -a experiments/webshop_pipeline.out
    "$PY" -c "
import json
r = json.load(open('$EVAL_RESULT'))
print('=== WebShop Summary ===')
for m, d in r.get('methods', {}).items():
    bm = d.get('buy_match_accuracy')
    bm_s = f'{bm:.1%}' if bm is not None else 'n/a'
    print(f'  {m:12s}: acc={d[\"accuracy\"]:.1%}  buy_match={bm_s:>6s}  '
          f'tok={d[\"avg_tokens\"]:.0f}  ({d[\"n_correct\"]}/{d[\"n_total\"]})')
" 2>&1 | tee -a experiments/webshop_pipeline.out
else
    echo "[$(date)] FAILURE. tail of eval log:" | tee -a experiments/webshop_pipeline.out
    tail -50 "$EVAL_OUT" | tee -a experiments/webshop_pipeline.out
fi
