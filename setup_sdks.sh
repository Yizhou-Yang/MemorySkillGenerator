#!/bin/bash
# SkillForge SDK Installer
# Installs Harbor + OpenHands into the harbor312 conda env.
# Run in background: nohup bash setup_sdks.sh &
#
# Requires: Python 3.12 conda env at /root/.conda/envs/harbor312
#           Network access to pypi.org and github.com
#           Docker (optional, for full agentic execution)

set -e
PYTHON=/root/.conda/envs/harbor312/bin/python
PIP="$PYTHON -m pip"

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

# Step 1: Harbor (Terminal-Bench evaluation framework)
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

# Verify
echo ""
echo "=== Verification ==="
$PYTHON -c "import harbor; print('harbor:', harbor.__version__)" 2>/dev/null || echo "harbor: NOT INSTALLED"
$PYTHON -c "import openhands; print('openhands: OK')" 2>/dev/null || echo "openhands: NOT INSTALLED"

echo ""
echo "=== Done ==="
echo "If SDKs are not installed, run this script again when network improves."
echo "In the meantime, SkillForge agents work in prompt-only mode."
