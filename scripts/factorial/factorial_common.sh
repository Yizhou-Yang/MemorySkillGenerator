cd /data/workspace/MemorySkillGenerator
set -a; source .env; set +a
source /tmp/hy3_env.sh
source /tmp/critic_env.sh
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 MEM0_TELEMETRY=False AMEM_PATH=/data/workspace/AgenticMemory
export C_POLICY=guarded METADATA_AUTHOR=critic
export ITER_MUTATE=1 ITER_FEEDBACK=self GEN_TEMPERATURE=0 RESUME=1
export TASK_CONCURRENCY=8 EXTERNAL_MEMS=""
export MUTATIONS_PATH=/data/workspace/MemorySkillGenerator/experiments_results/factorial/mutations.json
# GAIA2 needs its scenario directory or the loader silently finds 0 tasks and
# the arm "completes" in seconds.
export GAIA2_SCENARIO_DIR=/data/workspace/MemorySkillGenerator/.datasets/gaia2-cli-loaded
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export CODEBUDDY_MODEL=hy3 OPENAI_MODEL=hy3

run_arm () {  # $1=RESULTS_BASE $2=ARMS $3=ITER_CHAIN $4=TASK_LIMIT $5=BENCH (rest: env KEY=VAL)
  local base="$1" arms="$2" chain="$3" limit="$4" bench="${5:-gaia}"; shift 5
  # The mxzzz gateway flaps: a 503 window fails the critic startup gate with
  # zero work done. RESUME=1 makes relaunch monotone, so retry with backoff
  # rather than silently skipping the arm.
  local try rc
  for try in 1 2 3 4 5 6 7 8; do
    echo "=== [$base] arms=$arms chain=$chain n=$limit bench=$bench extra=$* try=$try $(date '+%m-%d %H:%M') ==="
    env "$@" RESULTS_BASE="$base" ARMS="$arms" BENCHMARKS="$bench" \
        ITER_CHAIN="$chain" TASK_LIMIT="$limit" \
        .venv/bin/python -u scripts/latest/latest_runner.py \
        >> "/tmp/run_${base//\//_}.log" 2>&1
    rc=$?
    echo "=== [$base] rc=$rc try=$try $(date '+%m-%d %H:%M') ==="
    [ "$rc" -eq 0 ] && return 0
    sleep 180
  done
  echo "=== [$base] GAVE UP after 8 tries $(date '+%m-%d %H:%M') ==="
  return 1
}
# --- 2026-08-20 critic choice, measured not guessed. On the shared gateway a
# 24-call burst at concurrency 8 gave gpt-5.6-sol 0/24 (all 503) and terra 17/24,
# and the startup gate kept declaring a live critic dead inside those windows.
# DeepSeek's own API answered 24/24 on the same probe, so critic and judge move
# there. These are new appendix experiments: what they need is one critic held
# fixed ACROSS their arms, which it is. The paper's frozen cells are untouched.
unset HY3_BASE_URL
export CRITIC_BASE_URL=https://api.deepseek.com
# Key is read from a file outside the repo; nothing secret is committed here.
export CRITIC_API_KEY=$(grep -oP '(?<=OPENAI_API_KEY=)[^ ]+' "${DS_ENV_FILE:-/tmp/ds_tau2_env.sh}" | tr -d '"')
export CRITIC_MODEL_ID=deepseek-v4-flash
export JUDGE_MODEL=deepseek-v4-flash
export JUDGE_BASE_URL=$CRITIC_BASE_URL
export JUDGE_API_KEY=$CRITIC_API_KEY
export C_CRITIC_TRIES=6 C_CRITIC_BACKOFF_S=5
# The gateway serves this critic in bursts (measured: terra 17/24 at concurrency
# 8, with the failures clustered). The startup gate probes 4 times over ~23s,
# which is shorter than an outage window, so a healthy critic gets declared dead
# and the arm exits before doing any work. Probe longer.
export PREFLIGHT_TRIES=12
