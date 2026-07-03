# TB2 → official harbor Terminus-2 integration (executable plan)

**Goal.** Replace our simplified "Terminus-2-style" loop with the **official
harbor harness + Terminus-2 reference agent**, keeping our A/B/C memory arms.
Result: TB2 absolute numbers become leaderboard-citable (reference line:
GPT-5 high + Terminus-2 = **42.5%**; the 80%+ leaderboard entries are custom
proprietary agents — never compare against those). Expected: GPT-5.5 row
~42–50%; hy3 row = its true level (~25–40%, unknown until measured).

## 0. Prereqs (server)

- Conda env `harbor312` exists (`/root/.conda/envs/harbor312/bin/python`).
- `pip show terminal-bench harbor` in that env; if missing:
  `pip install terminal-bench` (pulls harbor). Pin the version in this file
  after first install — leaderboard note: submissions may not modify
  timeouts/resources, so we do NOT touch task configs.
- Dataset: `harbor datasets download terminal-bench@2.0` (persist OUTSIDE
  /tmp, same lesson as gaia2: e.g. `<repo>/.datasets/terminal-bench-2`).
- Docker daemon + disk for images.

## 1. Model backend (any backbone, incl. hy3)

Terminus-2 calls the LLM via litellm. Point it at an OpenAI-compatible
endpoint (we already run one for hy3/CodeBuddy; see `.env.example` OPENAI_*):

```bash
export OPENAI_API_BASE=<internal-openai-compatible-endpoint>
export OPENAI_API_KEY=<key>
MODEL="openai/hy3-preview"      # litellm provider/model syntax
```

**Gate 1 (baseline sanity, no memory):**
```bash
harbor run -d terminal-bench/terminal-bench-2 -a terminus-2 -m "$MODEL" \
  -k 1 --n-tasks 10 --output-dir runs/tb2_smoke
```
Must produce nonzero scores and real agent transcripts (not prompt echoes).
If litellm rejects the endpoint, wrap it with a local litellm proxy.

## 2. Memory arms: subclass, do not fork

Create `scripts/latest/tb2_harbor_agent.py`:

- `class CuratedTerminus(Terminus2)` (import from `terminal_bench.agents`):
  override the method that renders the task instruction (in Terminus-2 this
  is where the task prompt is composed) to prepend our injected block:
  `mem.inject(task_dict)` → prefix `"## Relevant past solutions…"` when arm
  B/C, empty for A. Env `TB2_ARM=A|B|C` selects the arm; the agent reads the
  same `BenchmarkMemory` / `CuratedMemory` from `evomem_bridge` (task_id =
  harbor task id; chain = task_id ⇒ ITER_CHAIN semantics preserved by
  running k iterations per task, see §3).
- Register it as a custom agent (`harbor run -a
  scripts.latest.tb2_harbor_agent:CuratedTerminus`), harbor supports
  `module:Class` agent paths.
- **record()**: after each harbor task finishes, parse the run's
  `results.json` (per-task reward) and call `mem.record(task, result,
  score)` — same record-after-eval contract as latest_runner.

## 3. Iteration chains + trace

Harbor runs tasks independently; our chains = run the SAME task list K=3
times sequentially per arm (a thin driver loop, one harbor invocation per
iteration, memory object persisted in-process or serialized between
iterations). After each iteration, convert harbor's `results.json` into our
`trace.jsonl` rows (task_id, group, iteration, iter_total, score, em,
patch_injected, aug_len, code_rev) via a small
`scripts/latest/tb2_harbor_bridge.py` so `breakdown.py` / `analyze_results.py`
work unchanged.

## 4. Gates (in order)

1. Gate 1 smoke (above): 10 tasks, arm A, transcripts real.
2. `TB2_ARM=A` full n≈88, k=1 → compare vs the 42.5% reference given the
   backbone; sanity-check score distribution (no mass zeros).
3. `ITER_CHAIN=3` A/B/C run → `breakdown.py`: B/C injected n>0 on iters 1–2;
   error rate ≈0; then `analyze_results.py` C vs B.
4. Only then replace the TB2 column in the paper; appendix: swap
   "Terminus-2-style loop" for "official harbor Terminus-2 (version pinned),
   with a memory-prefix subclass"; cite the 42.5% reference line.

## 5. Effort estimate & fallback

~1–2 days server work (§1 hours; §2 the real work; §3 half day). If harbor's
agent-plugin API fights back, fallback = keep official harness for arm A
absolute numbers + our loop for A/B/C deltas, reported separately and
labeled — still kills the "6% harness" optics.

## Open questions (answer during §1)

- exact litellm provider string for the internal endpoint; context length cap
  for hy3 under Terminus-2's prompt size;
- harbor version pin + whether `--n-attempts/-k` semantics = trials (mean)
  — leaderboard uses k=5 mean; we report k=1 + our 3-iteration chains
  (different axis: chains are OUR protocol, documented in the paper).
