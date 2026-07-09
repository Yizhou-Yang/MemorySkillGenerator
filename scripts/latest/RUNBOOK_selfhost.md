# Self-host runbook — HY3-INT4 on 2×H200 for the benchmark sweep

Goal: spend the GPU window on **serving + computing**, not debugging. The pipeline
plumbing is already validated locally (mock endpoint); the only genuinely-new step
in the window is `vllm serve`.

**DeepSeek-V4-Pro is NOT self-hosted:** it is 1.6T params (~862 GB) → needs ~8×H200.
You already have its clean data via the API — keep it there. Self-host HY3 only.

---

## A. Before the window (do this early — downloads are the long pole)

1. **Stage HY3-INT4 weights on ceph** (persists across container recycle):
   ```
   # into your personal ceph dir, e.g. /apdcephfs/<you>/models/hy3-int4
   # get the safetensors INT4 checkpoint (AWQ/GPTQ/compressed-tensors)
   huggingface-cli download <hy3-int4-repo> --local-dir /apdcephfs/<you>/models/hy3-int4
   ```
   ~148 GB — hours depending on bandwidth. **Note the quant format** (awq / gptq /
   compressed-tensors) — you pass it as `QUANT=` below and it must match the checkpoint.

2. **Confirm the code + benchmark datasets are reachable on the node** (they live on
   the server ceph, not the laptop): the repo, and the GAIA/GAIA2/LoCoMo data +
   `GAIA2_SCENARIO_DIR`. If not, stage them on ceph too.

## B. In the window (run Claude Code INSIDE the container and I do these, or run by hand)

1. Repo + deps:
   ```
   cd <repo-on-ceph-or-clone>; git pull
   pip install vllm            # or the node's provided vLLM
   ```

2. **Serve** (survives IDE disconnect; loads ~148 GB from ceph → a few minutes):
   ```
   MODEL_PATH=/apdcephfs/<you>/models/hy3-int4 TP=2 QUANT=awq SERVED_NAME=hy3 \
     bash scripts/latest/deploy_vllm.sh
   ```
   Wait for `READY`. If it OOMs or errors: lower `MAX_LEN` (e.g. 16384), check `QUANT`
   matches the checkpoint, or `GPU_UTIL=0.90`. Log: `/tmp/vllm_hy3.log`.

3. **Point the pipeline at it + 2-task smoke (~5 min — proves data+model+pipeline align):**
   ```
   export LLM_PROVIDER=openrouter OPENROUTER_BASE_URL=http://localhost:8000/v1 \
          OPENROUTER_API_KEY=EMPTY CODEBUDDY_MODEL=hy3
   TASK_LIMIT=2 GAIA2_MAX_TURNS=20 RESULTS_BASE=latest_evolving \
     bash scripts/latest/run_all_benchmarks.sh hy3
   ```
   Check a trace row landed with a real score.

4. **Full run** (cost knobs already defaulted; results go under experiments_results):
   ```
   TASK_LIMIT=100 ITER_CHAIN=3 GAIA2_MAX_TURNS=20 RESULTS_BASE=latest_evolving \
     bash scripts/latest/run_all_benchmarks.sh hy3
   ```
   3-hour window realistically covers 1–2 benchmarks (e.g. GAIA2+GAIA); the rest in a
   later window (RESUME=1 continues where it stopped).

5. **Persist results to ceph continuously (auto-recycle = data loss otherwise):**
   ```
   # run in a second shell / tmux, loops every 2 min
   while true; do rsync -a experiments_results/ /apdcephfs/<you>/results/; sleep 120; done
   ```

## C. After / gate
   ```
   python scripts/latest/v2_gate.py hy3          # coverage / dose / marker check
   python scripts/latest/pooled_stats.py hy3 deepseek-v4-pro   # pool with the API anchor
   ```

## Notes
- **Gate before trusting:** the previous hy3 data was contaminated (identical B-arm
  across dirs); this is a fresh clean run — confirm `inject n>0` per bench with
  `breakdown.py` and that group keys are canonical.
- Keep the server up the whole window (one load). Don't restart per benchmark.
- If HY3-INT4 quality looks off vs the API, note it — INT4 on a 295B MoE trades some
  quality; 4×H200 FP8 is the higher-fidelity option for a later window.
