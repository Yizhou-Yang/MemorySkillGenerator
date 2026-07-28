#!/bin/bash
# SkillForge SDK Installer
# Installs Harbor + OpenHands + ARE (meta-agents-research-environments)
# into the harbor312 conda env, then converts the GAIA2 HF dataset to the
# CLI directory layout that the official harbor ARE loader expects.
#
# Run in background: nohup bash setup_sdks.sh &
#
# Requires: Python 3.12 conda env (harbor312) on the PERSISTENT ceph mount:
#             /apdcephfs_hzlf/share_1227201/samzxge/miniconda3/envs/harbor312
#           Network access to pypi.org and github.com
#           Docker (optional, for full agentic execution)
#
# NOTE: This env lives on ceph, NOT the container system disk. A container/
# image reset (which zeroes the system disk) does NOT wipe it — as long as
# the ceph mount is re-attached, `harbor312` is still usable. This script is
# only needed if the env is missing (e.g. fresh ceph or manual cleanup).

set -e
PYTHON=/apdcephfs_hzlf/share_1227201/samzxge/miniconda3/envs/harbor312/bin/python
PIP="$PYTHON -m pip"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "=== SkillForge SDK Installer ==="
echo "Target: $PYTHON"
echo "Time:   $(date)"
echo ""

install_with_retry() {
    local pkg="$1"
    local max_retries=5
    local retry=0
    while [ $retry -lt $max_retries ]; do
        echo "[$(date +%H:%M:%S)] Installing $pkg (attempt $((retry+1))/$max_retries)..."
        if $PIP install --default-timeout=120 "$pkg" 2>&1; then
            echo "[$(date +%H:%M:%S)] $pkg installed successfully"
            return 0
        fi
        retry=$((retry+1))
        echo "[$(date +%H:%M:%S)] Retry $retry/$max_retries after 30s..."
        sleep 30
    done
    echo "[$(date +%H:%M:%S)] FAILED to install $pkg after $max_retries attempts"
    return 1
}

# Step 1: Harbor (container evaluation framework)
echo ""
echo "--- Harbor ---"
install_with_retry "git+https://github.com/harbor-framework/harbor.git" || \
    echo "Harbor NOT installed (network issue). Prompt-only mode will be used."

# Step 2: OpenHands (code engineering agent SDK)
# NOTE: Use PyPI wheel (not git source) to avoid tree-sitter-language-pack
# build failure when rust/cargo toolchain is unavailable.
echo ""
echo "--- OpenHands ---"
install_with_retry "openhands" || \
    echo "OpenHands NOT installed (network issue). Prompt-only mode will be used."

# Step 3: ARE — meta-agents-research-environments
# Provides `are` (load_scenario) used by are_integration.py to load the GAIA2
# CLI scenarios. REQUIRED for the official harbor312 GAIA2 load path.
echo ""
echo "--- ARE (meta-agents-research-environments) ---"
install_with_retry "meta-agents-research-environments" || \
    echo "ARE NOT installed (network issue). GAIA2 load will fail until fixed."

# Verify
echo ""
echo "=== Verification ==="
$PYTHON -c "import harbor; print('harbor:', harbor.__version__)" 2>/dev/null || echo "harbor: NOT INSTALLED"
$PYTHON -c "import openhands; print('openhands: OK')" 2>/dev/null || echo "openhands: NOT INSTALLED"
$PYTHON -c "import are; print('are (ARE):', getattr(are, '__version__', 'OK'))" 2>/dev/null || echo "are (ARE): NOT INSTALLED"

# Step 4: GAIA2 dataset — convert HF parquet -> CLI directory (idempotent)
# Source parquet lives in .datasets/gaia2-cli (HF format). The official loader
# reads the CLI layout at .datasets/gaia2-cli-loaded. Skip if already present.
echo ""
echo "--- GAIA2 dataset (HF -> CLI) ---"
CLI_OUT="$REPO_ROOT/.datasets/gaia2-cli-loaded"
if [ -d "$CLI_OUT" ] && [ -n "$(find "$CLI_OUT" -name scenario.json 2>/dev/null | head -1)" ]; then
    echo "GAIA2 CLI dataset already present at $CLI_OUT — skipping conversion."
else
    echo "Converting GAIA2 HF parquet -> CLI layout..."
    $PYTHON "$REPO_ROOT/scripts/latest/convert_gaia2_hf_to_cli.py" \
        --src "$REPO_ROOT/.datasets/gaia2-cli" \
        --out "$CLI_OUT" || \
        echo "GAIA2 conversion FAILED (check .datasets/gaia2-cli source)."
fi

echo ""
echo "=== Done ==="
echo "If SDKs are not installed, run this script again when network improves."
echo "In the meantime, SkillForge agents work in prompt-only mode."
