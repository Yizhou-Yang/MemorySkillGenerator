#!/bin/bash
# Run official Terminus-2 via terminal-bench harness with CodeBuddy proxy.
#
# This script:
#   1. Starts the CodeBuddy OAI proxy (if not already running)
#   2. Runs terminal-bench with the specified arm/model/tasks
#
# Usage:
#   # Smoke test (1 task, arm A):
#   bash scripts/latest/run_tb2_official.sh --arm A --n-tasks 1
#
#   # Full run (all tasks, arm B, 3 iterations):
#   bash scripts/latest/run_tb2_official.sh --arm B --iters 3
#
#   # Background (recommended for full runs):
#   nohup bash scripts/latest/run_tb2_official.sh --arm C --iters 3 \
#       > logs/tb2_official_C.log 2>&1 &

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SKILLFORGE_PYTHON="/root/.conda/envs/skillforge/bin/python"
HARBOR_PYTHON="/root/.conda/envs/harbor312/bin/python"
PROXY_PORT=8741
PROXY_SCRIPT="$SCRIPT_DIR/codebuddy_oai_proxy.py"
BRIDGE_SCRIPT="$SCRIPT_DIR/tb2_harbor_bridge.py"

echo "[run_tb2_official] Starting at $(date)"
echo "[run_tb2_official] Project root: $PROJECT_ROOT"

# --- Step 1: Ensure proxy is running ---
if curl -s "http://localhost:$PROXY_PORT/health" > /dev/null 2>&1; then
    echo "[run_tb2_official] Proxy already running on port $PROXY_PORT"
else
    echo "[run_tb2_official] Starting CodeBuddy OAI proxy on port $PROXY_PORT..."
    mkdir -p "$PROJECT_ROOT/logs"
    nohup "$SKILLFORGE_PYTHON" "$PROXY_SCRIPT" \
        > "$PROJECT_ROOT/logs/codebuddy_proxy.log" 2>&1 &
    PROXY_PID=$!
    echo "[run_tb2_official] Proxy PID: $PROXY_PID"
    sleep 3
    if ! curl -s "http://localhost:$PROXY_PORT/health" > /dev/null 2>&1; then
        echo "[run_tb2_official] ERROR: Proxy failed to start. Check logs/codebuddy_proxy.log"
        exit 1
    fi
    echo "[run_tb2_official] Proxy started successfully"
fi

# --- Step 2: Run the bridge ---
export OPENAI_API_BASE="http://localhost:$PROXY_PORT/v1"
export OPENAI_API_KEY="dummy"
export CODEBUDDY_API_KEY="${CODEBUDDY_API_KEY:-$(grep CODEBUDDY_API_KEY "$PROJECT_ROOT/.env" | cut -d= -f2)}"
export CODEBUDDY_INTERNET_ENVIRONMENT="${CODEBUDDY_INTERNET_ENVIRONMENT:-ioa}"

echo "[run_tb2_official] OPENAI_API_BASE=$OPENAI_API_BASE"
echo "[run_tb2_official] Running bridge with args: $@"
echo ""

exec "$HARBOR_PYTHON" "$BRIDGE_SCRIPT" "$@"
