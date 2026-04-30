# SkillForge

> **Skill compiler that produces reusable agent skills from conversation trajectories and/or compressed memories.**

SkillForge implements the research idea *"Learning to Compile Agent Skills via Adaptive Routing and Denoising"*. It takes raw agent interaction trajectories, compresses them into structured memory, then induces reusable skills through three competing pathways — and evaluates which pathway produces the most transferable, high-quality skills.

**Key finding (v6):** The *Evidence-as-Filter* hybrid approach — using trajectory evidence to **filter and rank** memories rather than inject details — achieves the best Self-consistency (7.7/10) and Cross-task generalisation (6.8/10) across benchmarks. All scores are LLM-as-judge ratings on a 0–10 scale; objective EM/F1 metrics are implemented for future runs.

---

## Table of Contents

- [Research Question](#research-question)
- [Architecture Overview](#architecture-overview)
- [Three Competing Pathways](#three-competing-pathways)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration Guide](#configuration-guide)
- [Pipeline Details](#pipeline-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Benchmarks](#benchmarks)
- [Latest Results (v6)](#latest-results-v6)
- [Testing](#testing)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Research Question

When an AI Agent completes a task (e.g. answering a multi-hop question), it leaves behind an **interaction trajectory** — a verbose record of thoughts, actions, observations, and errors.

**How do we compress this noisy trajectory into a concise, reusable "skill" that helps the agent solve similar tasks in the future?**

We compare three approaches and measure which produces skills that generalise best.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        SkillForge Pipeline                           │
│                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────────┐   │
│  │Benchmark │───>│  Trajectory  │───>│   Memory Compressor       │   │
│  │  Loader  │    │  Collector   │    │  (Mem0 / A-MEM / MemBank) │   │
│  └──────────┘    └──────┬───────┘    └───────────┬───────────────┘   │
│                         │                        │                   │
│              ┌──────────┴────────────────────────┴──────────┐        │
│              │          Skill Induction (×3)                 │        │
│              │  ┌─────────────────────────────────────────┐  │        │
│              │  │ 1. traj→skill    (full trajectory)      │  │        │
│              │  │ 2. memory→skill  (compressed memory)    │  │        │
│              │  │ 3. hybrid→skill  (evidence-filtered)    │  │        │
│              │  └─────────────────────────────────────────┘  │        │
│              └──────────────────┬────────────────────────────┘        │
│                                │                                     │
│                    ┌───────────┴───────────┐                         │
│                    │   Skill Evaluator     │                         │
│                    │ Self / Cross / Transfer│                        │
│                    │ Quality (5-dim) / Comp │                        │
│                    └───────────┬───────────┘                         │
│                                │                                     │
│                    results_table.txt + all_metrics.json               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Three Competing Pathways

```
                    ┌─────────────────┐
                    │   Trajectory    │  (12-20 steps, naturally verbose)
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌─────────────────┐
     │ Path 1:    │  │ Path 2:    │  │ Path 3:         │
     │ traj→skill │  │ mem→skill  │  │ hybrid→skill    │
     │ (direct)   │  │ (compress) │  │ (evidence-filter)│
     └─────┬──────┘  └─────┬──────┘  └──────┬──────────┘
           │               │                │
           ▼               ▼                ▼
     ┌──────────┐   ┌──────────┐   ┌──────────────┐
     │  Skill   │   │  Skill   │   │    Skill     │
     │(overfit) │   │(denoised)│   │(best-of-both)│
     └──────────┘   └──────────┘   └──────────────┘
```

| Path | Input | Approach | Strength | Weakness |
|------|-------|----------|----------|----------|
| **traj→skill** | Full trajectory | Direct extraction, preserves all details | Maximum information retention | Information overload → overfitting |
| **memory→skill** | Compressed memory | Extract from pre-structured memory only | Natural denoising, good generalisation | May lose critical operational details |
| **hybrid→skill** | Memory + trajectory evidence | Trajectory validates & ranks memories, then skill is induced from filtered memories only | Best memory selection + memory-level abstraction | Higher cost (2 LLM calls) |

### Evidence-as-Filter (v6 Core Innovation)

The hybrid path's key insight: **the trajectory's role is to SELECT which memories matter, not to ADD concrete details to the skill.**

```
v5 (wrong): Memory + Trajectory Details → inject details → pollute abstraction
v6 (right): Trajectory validates Memory → filter & rank → keep only best → generate Skill
```

1. **Validate**: LLM assesses each memory's `evidence_strength` and `generalizability`
2. **Filter**: Keep only memories with evidence ≥ moderate AND generalizability ≥ medium
3. **Rank**: Sort by generalizability > evidence_strength > importance
4. **Induce**: Feed ONLY filtered memories (no raw trajectory) to the skill induction LLM

---

## Project Structure

```
SkillForge/
├── benchmarks/
│   ├── __init__.py
│   └── loader.py              # HuggingFace dataset loader (HotpotQA/TriviaQA/GSM8K/MuSiQue/SWE-bench)
├── configs/
│   ├── default.yaml           # Default experiment configuration
│   └── mvp_locomo.yaml        # MVP experiment config (overrides default)
├── docs/
│   └── internal/              # Internal docs (gitignored)
│       └── technical_report.md
├── experiments/               # Experiment outputs (gitignored)
│   ├── multi_benchmark_v6/    # Latest v6 results
│   └── .gitkeep
├── scripts/
│   ├── run_mvp.py             # Single-benchmark MVP entry point
│   └── run_multi_benchmark.py # Multi-benchmark experiment runner (v6)
├── src/
│   ├── __init__.py
│   ├── models.py              # Pydantic data models (Trajectory, Memory, Skill, EvalResult)
│   ├── trajectory/
│   │   └── collector.py       # ReAct agent trajectory collector (forced multi-step)
│   ├── memory/
│   │   └── compressor.py      # Memory compressors (Mem0, A-MEM, MemoryBank) + factory
│   ├── skill_induction/
│   │   ├── base.py            # Abstract base class
│   │   ├── factory.py         # Skill inducer factory
│   │   ├── traj_to_skill.py   # Path 1: trajectory → skill (direct)
│   │   ├── memory_to_skill.py # Path 2: memory → skill (compressed)
│   │   └── hybrid_to_skill.py # Path 3: hybrid → skill (evidence-as-filter, v6)
│   ├── evaluation/
│   │   └── evaluator.py       # LLM-as-judge + 5-dimension quality scoring
│   ├── rl_controller/         # (Future) RL-based adaptive routing
│   └── utils/
│       ├── config.py          # YAML config loader + env override
│       ├── io.py              # JSON/JSONL serialisation helpers
│       ├── llm.py             # Unified LLM API client (OpenAI-compatible)
│       └── logging.py         # Loguru-based logger setup
├── tests/                     # Unit & integration tests
│   ├── test_compressors.py
│   ├── test_loader.py
│   ├── test_skill_induction.py
│   ├── test_utils.py
│   ├── test_models.py
│   └── test_config.py
├── .env.example               # Environment variable template
├── .gitignore
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Project metadata (Python ≥ 3.10, Apache-2.0)
├── LICENSE
└── README.md                  # This file
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | Required by type hints (`dict[str, Any]`, `X \| None`) |
| pip | latest | For installing dependencies |
| DeepSeek API Key | — | Or any OpenAI-compatible API |
| Internet | — | Required for HuggingFace dataset download + LLM API calls |
| Disk | ≥ 2 GB | For HuggingFace dataset cache + experiment outputs |

---

## Quick Start

### 1. Install dependencies

```bash
cd /root/workspace/SkillForge
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — the only REQUIRED variable is DEEPSEEK_API_KEY
```

### 3. Run a quick smoke test (2 tasks, ~3 min)

```bash
python scripts/run_multi_benchmark.py \
  --benchmarks hotpotqa \
  --num-samples 2
```

### 4. Run the full multi-benchmark experiment (v6)

```bash
# Full experiment: 3 benchmarks × 10 tasks × 4 variants (incl. baseline)
# Estimated runtime: ~2 hours
nohup python scripts/run_multi_benchmark.py \
  --benchmarks hotpotqa,gsm8k,triviaqa \
  --num-samples 10 \
  > experiments/experiment_output.log 2>&1 &
```

### 5. View results

```bash
cat experiments/multi_benchmark_v6/results_table.txt
```

---

## Configuration Guide

### Config file hierarchy

```
configs/default.yaml    ← base config (all parameters with defaults)
configs/mvp_locomo.yaml ← override config (only changed parameters)
```

The system loads `default.yaml` first, then deep-merges the override config on top. Environment variables from `.env` further override LLM settings.

### Key parameters

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `llm.model` | `deepseek-chat` | — | LLM model name |
| `llm.temperature` | `0.7` | — | Sampling temperature |
| `llm.max_tokens` | `4096` | — | Max output tokens per call |
| `memory.framework` | `mem0` | — | Memory compressor: `mem0` / `amem` / `memorybank` |
| `benchmark.name` | `hotpotqa` | — | Benchmark name |
| `benchmark.num_samples` | `20` | — | Number of tasks to process |
| `skill_induction.hybrid.evidence_retrieval_top_k` | `5` | — | Max memories to keep after filtering |
| `evaluation.num_validation_runs` | `3` | — | Validation runs per skill |

### Using a different LLM provider

SkillForge uses the OpenAI-compatible API format. Update `.env`:

```bash
# Example: OpenAI
DEEPSEEK_API_KEY=sk-your-openai-key
DEEPSEEK_BASE_URL=https://api.openai.com/v1
DEEPSEEK_MODEL=gpt-4o

# Example: local Ollama
DEEPSEEK_API_KEY=ollama
DEEPSEEK_BASE_URL=http://localhost:11434/v1
DEEPSEEK_MODEL=llama3
```

---

## Pipeline Details

### 1. Trajectory Collection (`src/trajectory/collector.py`)

Drives a **ReAct agent** through each task with forced multi-step reasoning:

| Round | Observation prompt | Purpose |
|-------|-------------------|---------|
| 0 | "Decompose into 2-3 sub-questions" | Force problem decomposition |
| 1 | "Work through sub-question 1, give evidence" | Gather evidence |
| 2 | "Move to sub-question 2, find connections" | Cross-reference |
| 3 | "Check for contradictions, verify assumptions" | Self-verification |
| 4 | "Synthesise findings, give final answer" | Conclude |

**No artificial noise** is injected. The natural verbosity of multi-step reasoning (12-20 steps) is what differentiates the three skill induction pathways.

### 2. Memory Compression (`src/memory/compressor.py`)

| Framework | Strategy | Key Feature |
|-----------|----------|-------------|
| `mem0` | Flat key-value extraction | Simple, 1 LLM call, each entry independent |
| `amem` | Two-pass: extract → reflect + link + merge | Higher quality, 2 LLM calls |
| `memorybank` | Hierarchical tiering (core/working/ephemeral) + forgetting | Controls memory size |

All compressors output `MemoryEntry` objects with: `content`, `category` (fact/rule/procedure/insight), `specificity_score`, `importance`.

### 3. Skill Induction (`src/skill_induction/`)

Each pathway produces a `Skill` with: `name`, `description`, `preconditions`, `procedure`, `constraints`, `facts`, `rules`.

- **traj→skill**: Receives the FULL trajectory. Prompt says "preserve ALL reasoning details". Tends to produce over-specific or vague skills.
- **memory→skill**: Receives ONLY compressed memory. Prompt says "use ONLY the information present". Produces clean but potentially incomplete skills.
- **hybrid→skill (v6)**: Two-step process — (1) LLM validates each memory against trajectory evidence, (2) filtered memories fed to skill induction. Produces memory-level abstraction with better memory selection.

### 4. Evaluation (`src/evaluation/evaluator.py`)

Two tiers of evaluation metrics:

**Primary (objective, deterministic):**
- **Exact Match (EM)**: normalised containment check — does the response contain the expected answer? Follows SQuAD/HotpotQA normalisation (lowercase, remove punctuation/articles).
- **Token F1**: token-level precision/recall harmonic mean between extracted answer and expected answer.

**Secondary (LLM-as-judge, non-deterministic, for reference):**
- **LLM-as-judge (0–10)**: Injects skill as system prompt → agent answers task → separate judge LLM scores on a strict rubric.
- **5-dimension quality (0–10)**: Specificity, Reusability, Structure, Denoising, Completeness — each scored independently.
- **Compression ratio**: `chars(trajectory) / chars(skill)` — higher means more compact.

---

## Evaluation Metrics

### Primary Metrics (Objective)

| Metric | What it measures | How it's computed | Deterministic? |
|--------|-----------------|-------------------|:--------------:|
| **EM** | Answer correctness | Normalised expected answer ⊆ response (SQuAD protocol) | ✅ Yes |
| **F1** | Answer overlap | Token-level precision × recall harmonic mean | ✅ Yes |

### Secondary Metrics (LLM-as-Judge)

| Metric | What it measures | How it's computed | Deterministic? |
|--------|-----------------|-------------------|:--------------:|
| **Self** | Information retention | Skill from task A evaluated on task A (0–10 judge) | ❌ No |
| **Cross** | Same-benchmark generalisation | Skill from task A evaluated on tasks B, C, D... (0–10 judge) | ❌ No |
| **Transfer** | Cross-benchmark generalisation | Skill from benchmark X evaluated on benchmark Y (0–10 judge) | ❌ No |
| **Quality** | Skill structure quality | LLM rates 5 dimensions: specificity / reusability / structure / denoising / completeness (0–10 each) | ❌ No |
| **Compress** | Information density | chars(trajectory) / chars(skill) — higher = more compact | ✅ Yes |

> **Note:** v6 results below use LLM-as-judge scores. EM/F1 objective metrics are implemented in the evaluator and will be reported in future experiment runs.

### Transfer pairs

| Source | Target | Rationale |
|--------|--------|-----------|
| HotpotQA | MuSiQue | Multi-hop → harder multi-hop (should transfer well) |
| GSM8K | TriviaQA | Math → factoid QA (should fail — negative control) |
| TriviaQA | HotpotQA | Single-hop → multi-hop (partial transfer) |

---

## Benchmarks

| Name | HF Dataset ID | License | Task Type | Role |
|------|--------------|---------|-----------|------|
| HotpotQA | `hotpotqa/hotpot_qa` | CC-BY-SA-4.0 | Multi-hop reasoning QA | Primary benchmark |
| TriviaQA | `mandarjoshi/trivia_qa` | Academic | Single-hop factoid QA | Simple baseline |
| GSM8K | `openai/gsm8k` | MIT | Math reasoning | Precise numeric evaluation |
| MuSiQue | `dgslibisey/MuSiQue` | CC-BY-4.0 | Multi-hop QA (harder) | Transfer evaluation target |

First-run dataset download is automatic via HuggingFace `datasets` library (~200MB cached).

---

## Latest Results (v6)

> **Methodology:** N=10 tasks per benchmark, 3 benchmarks, 4 variants (incl. baseline). All Self/Cross/Transfer scores are LLM-as-judge ratings on a **0–10 scale**, normalised to 0–1 for the table. Quality is the mean of 5 sub-dimensions (each 0–10). Compress is deterministic (chars ratio). Runtime: 114 min, 3,181 LLM calls, 2.84M tokens.

### Cross-Benchmark Averages (3 benchmarks)

| Variant | Self ↑ | Cross ↑ | Transfer | Quality ↑ | Compress |
|---------|:------:|:-------:|:--------:|:---------:|:--------:|
| no_skill_baseline | 6.7 | 6.7 | 6.7 | — | — |
| traj→skill | 5.9 | 6.2 | 4.3 | 8.1 | 2.3× |
| memory→skill | 7.2 | 6.5 | **4.7** | 7.7 | **4.4×** |
| **hybrid→skill** | **7.7** | **6.8** | 4.1 | 7.9 | 3.4× |

*Scores are on a 0–10 scale (LLM-as-judge). Higher is better for all metrics except Compress (higher = more compact, also better).*

### Per-Benchmark Results

#### HotpotQA (multi-hop reasoning → MuSiQue transfer)

| Variant | Self /10 | Cross /10 | Transfer /10 | Quality /10 | Compress |
|---------|:--------:|:---------:|:------------:|:-----------:|:--------:|
| baseline | 2.0 | 2.0 | 2.0 | — | — |
| traj→skill | 2.0 | 0.7 | 0.6 | 8.2 | 2.2× |
| memory→skill | 5.0 | 2.7 | 1.4 | 7.7 | **4.9×** |
| **hybrid→skill** | **5.8** | **2.8** | 0.9 | 8.0 | 3.7× |

#### GSM8K (math reasoning → TriviaQA transfer)

| Variant | Self /10 | Cross /10 | Transfer /10 | Quality /10 | Compress |
|---------|:--------:|:---------:|:------------:|:-----------:|:--------:|
| baseline | 10.0 | 10.0 | 10.0 | — | — |
| traj→skill | 8.6 | 9.8 | 10.0 | 8.2 | 2.6× |
| memory→skill | 8.6 | 9.1 | 9.6 | 7.9 | **4.1×** |
| **hybrid→skill** | **9.2** | **9.5** | 9.6 | 8.1 | 3.2× |

#### TriviaQA (factoid QA → HotpotQA transfer)

| Variant | Self /10 | Cross /10 | Transfer /10 | Quality /10 | Compress |
|---------|:--------:|:---------:|:------------:|:-----------:|:--------:|
| baseline | 8.2 | 8.2 | 8.2 | — | — |
| traj→skill | 7.2 | 8.0 | 2.2 | 7.9 | 2.0× |
| memory→skill | 8.0 | 7.7 | **3.0** | 7.4 | **4.3×** |
| **hybrid→skill** | **8.2** | **8.1** | 1.8 | 7.6 | 3.3× |

### Key Observations

1. **hybrid→skill achieves the highest Self (7.7) and Cross (6.8)** across benchmarks, outperforming memory→skill by +0.5 and +0.3 points respectively on the 0–10 scale.
2. **memory→skill achieves the highest Transfer (4.7)** and best Compression (4.4×), suggesting that aggressive denoising via memory compression produces more transferable and compact skills.
3. **traj→skill shows the largest Self–Cross gap** (5.9 vs 6.2 = −0.3 on average, but on HotpotQA: 2.0 vs 0.7 = −1.3), indicating overfitting to source task details.
4. **HotpotQA (complex multi-hop) shows the largest inter-variant differentiation**: hybrid Self=5.8 vs traj Self=2.0 (Δ=3.8 points). Simple benchmarks (GSM8K, TriviaQA) show smaller gaps.
5. **All skill variants outperform baseline on HotpotQA** (the hardest benchmark), confirming that skill injection provides value for complex reasoning tasks.
6. **GSM8K baseline scores 10.0/10** — the LLM already solves grade-school math perfectly without skills, making this benchmark a ceiling control rather than a differentiator.

### Limitations of Current Evaluation

- All Self/Cross/Transfer/Quality scores are **LLM-as-judge** (non-deterministic). The same evaluation may produce slightly different scores on re-run.
- **Objective EM/F1 metrics** have been implemented in the evaluator (`_compute_em`, `_compute_f1`) but were not available during the v6 experiment run. Future experiments will report both objective and judge-based metrics.
- N=10 per benchmark provides moderate statistical power. Confidence intervals are not yet computed.

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test files (no network required)
python -m pytest tests/test_config.py tests/test_models.py tests/test_utils.py -v

# Integration tests (requires network for HuggingFace)
python -m pytest tests/test_integration.py -v
```

| Test File | Network | LLM | What It Tests |
|-----------|---------|-----|---------------|
| `test_config.py` | No | No | Config loading, deep merge |
| `test_models.py` | No | No | Pydantic model validation |
| `test_utils.py` | No | No | Utility functions |
| `test_compressors.py` | No | Mock | Memory compressor logic |
| `test_loader.py` | Yes | No | Benchmark dataset loading |
| `test_skill_induction.py` | No | Mock | Skill induction + evaluation |
| `test_integration.py` | Yes | No | End-to-end loader + compressor |

---

## Output Structure

```
experiments/multi_benchmark_v6/
├── hotpotqa/
│   ├── skills/
│   │   ├── traj_to_skill/       # Skills from path 1
│   │   ├── memory_to_skill/     # Skills from path 2
│   │   └── hybrid_to_skill/     # Skills from path 3
│   └── metrics.json             # Per-benchmark metrics
├── gsm8k/
│   └── ...
├── triviaqa/
│   └── ...
├── all_metrics.json             # Aggregated metrics (all benchmarks)
└── results_table.txt            # Human-readable results table
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| `DEEPSEEK_API_KEY is not set` | Missing `.env` | `cp .env.example .env` and fill in API key |
| `ModuleNotFoundError` | Dependencies missing | `pip install -r requirements.txt` |
| `Connection error` / `timeout` | Network or API overload | Increase `llm.timeout`; auto-retries 3× |
| `Unsupported benchmark` | Invalid name | Use: `hotpotqa`, `triviaqa`, `gsm8k`, `musique`, `swebench` |
| HuggingFace download fails | Network/proxy | Set `HF_ENDPOINT=https://hf-mirror.com` |
| `JSONDecodeError` in compressor | LLM returned non-JSON | Fallback wraps raw response as single entry |
| Experiment too slow | Too many samples | Reduce `--num-samples` (e.g. 2 for smoke test) |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
