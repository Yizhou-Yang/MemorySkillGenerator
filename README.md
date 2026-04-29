# SkillForge

> **Skill compiler which produces reusable agent skills from conversation trajectories and/or compressed memories.**

SkillForge implements the paper idea *"Learning to Compile Agent Skills via Adaptive Routing and Denoising"*. It takes raw agent interaction trajectories, compresses them into structured memory, then induces reusable skills through three competing pathways. The best pathway is selected via evaluation.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start (5 Steps to Run)](#quick-start-5-steps-to-run)
- [Configuration Guide](#configuration-guide)
- [Pipeline Details](#pipeline-details)
- [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Output Structure](#output-structure)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        SkillForge Pipeline                      │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │Benchmark │───>│  Trajectory  │───>│   Memory Compressor   │  │
│  │  Loader  │    │  Collector   │    │  (Mem0/A-MEM/MemBank) │  │
│  └──────────┘    └──────┬───────┘    └───────────┬───────────┘  │
│                         │                        │              │
│                         v                        v              │
│              ┌──────────────────────────────┐                   │
│              │     Skill Induction (x3)     │                   │
│              │  ┌────────────────────────┐  │                   │
│              │  │ 1. traj_to_skill       │  │  Trajectory only  │
│              │  │ 2. memory_to_skill     │  │  Memory only      │
│              │  │ 3. hybrid_to_skill     │  │  Both combined    │
│              │  └────────────────────────┘  │                   │
│              └──────────────┬───────────────┘                   │
│                             │                                   │
│                             v                                   │
│                    ┌────────────────┐                            │
│                    │ Skill Evaluator│                            │
│                    │ (compare x3)  │                             │
│                    └───────┬────────┘                            │
│                            │                                    │
│                            v                                    │
│                   comparison.json                               │
│                   (winner + metrics)                             │
└─────────────────────────────────────────────────────────────────┘
```

**Core flow:**
1. **Benchmark Loader** — loads tasks from HuggingFace datasets (HotpotQA / SWE-bench)
2. **Trajectory Collector** — drives a ReAct agent through each task, recording every thought/action/observation step
3. **Memory Compressor** — compresses raw trajectories into structured memory entries (3 strategies: Mem0 flat extraction, A-MEM agentic reflection, MemoryBank hierarchical tiering)
4. **Skill Induction** — generates reusable skills from trajectories and/or memory via 3 competing pathways
5. **Skill Evaluator** — validates each skill on tasks, computes success rate / compression ratio, and compares all 3 variants

---

## Project Structure

```
SkillForge/
├── configs/
│   ├── default.yaml          # Default experiment configuration (all parameters)
│   └── mvp_locomo.yaml       # MVP experiment config (overrides default)
├── benchmarks/
│   ├── __init__.py
│   └── loader.py             # HuggingFace dataset loader (HotpotQA, SWE-bench, HotpotQA-hard)
├── src/
│   ├── __init__.py
│   ├── models.py             # Pydantic data models (Trajectory, Memory, Skill, EvalResult)
│   ├── trajectory/
│   │   └── collector.py      # ReAct agent trajectory collector
│   ├── memory/
│   │   └── compressor.py     # Memory compressors (Mem0, A-MEM, MemoryBank) + factory
│   ├── skill_induction/
│   │   ├── base.py           # Abstract base class for skill inducers
│   │   ├── factory.py        # Skill inducer factory
│   │   ├── traj_to_skill.py  # Variant 1: trajectory → skill (direct)
│   │   ├── memory_to_skill.py# Variant 2: memory → skill (compressed)
│   │   └── hybrid_to_skill.py# Variant 3: trajectory + memory → skill (hybrid)
│   ├── evaluation/
│   │   └── evaluator.py      # Skill quality evaluator + variant comparison
│   ├── rl_controller/        # (Future) RL-based adaptive routing
│   └── utils/
│       ├── config.py         # YAML config loader + env override
│       ├── io.py             # JSON/JSONL serialisation helpers
│       ├── llm.py            # Unified LLM API client (OpenAI-compatible)
│       └── logging.py        # Loguru-based logger setup
├── scripts/
│   └── run_mvp.py            # Main experiment entry point
├── tests/
│   ├── test_config.py        # Config loading tests
│   ├── test_models.py        # Data model tests
│   └── test_integration.py   # Benchmark loader + compressor factory tests
├── experiments/               # Output directory (auto-created, gitignored)
├── .env.example              # Environment variable template
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata (Python ≥ 3.10, MIT license)
└── README.md                 # This file
```

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.10 | Required by type hints (`dict[str, Any]`, `X \| None`) |
| pip | latest | For installing dependencies |
| DeepSeek API Key | — | Or any OpenAI-compatible API (HunyuanLLM, etc.) |
| Internet | — | Required for HuggingFace dataset download + LLM API calls |
| Disk | ≥ 2 GB | For HuggingFace dataset cache + experiment outputs |

---

## Quick Start (5 Steps to Run)

### Step 1: Clone and enter the project

```bash
cd /root/workspace/SkillForge
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If `sentence-transformers` or `torch` installation is slow, you can skip them for the MVP experiment (they are only needed for local embedding, not for the core pipeline):
> ```bash
> pip install openai tiktoken pydantic pyyaml python-dotenv loguru rich tqdm tenacity jinja2 datasets rouge-score pandas numpy langgraph langchain langchain-openai
> ```

### Step 3: Configure environment variables

```bash
cp .env.example .env
```

Then edit `.env` with your actual API key:

```bash
# .env file content — fill in your actual values:
DEEPSEEK_API_KEY=sk-your-actual-deepseek-api-key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DEVICE=cpu
MEM0_VECTOR_STORE=chromadb
MEM0_COLLECTION_NAME=skillforge_memory
EXPERIMENT_DIR=./experiments
LOG_LEVEL=INFO
```

**Key:** The only **required** variable is `DEEPSEEK_API_KEY`. All others have working defaults.

### Step 4: Verify configuration (dry-run)

```bash
python scripts/run_mvp.py --config mvp_locomo --dry-run
```

Expected output:
```
SkillForge MVP experiment started: config=mvp_locomo
Config:
{
  "llm": {"model": "deepseek-chat", "temperature": 0.7, ...},
  "benchmark": {"name": "hotpotqa", "num_samples": 20},
  ...
}
Dry-run mode — exiting
```

If you see the config printed without errors, the setup is correct.

### Step 5: Run the full experiment

```bash
python scripts/run_mvp.py --config mvp_locomo
```

This will:
1. Download HotpotQA dataset from HuggingFace (first run only, ~200MB cached)
2. Load 20 sample tasks
3. For each task: collect trajectory → compress memory → induce 3 skill variants → evaluate
4. Save all results to `experiments/mvp_locomo/`
5. Print final comparison of the 3 variants

**Estimated runtime:** ~15-30 minutes (depends on API latency, 20 tasks × 3 variants × multiple LLM calls each)

---

## Configuration Guide

### Config file hierarchy

```
configs/default.yaml    ← base config (all parameters with defaults)
configs/mvp_locomo.yaml ← override config (only changed parameters)
```

The system loads `default.yaml` first, then deep-merges the override config on top. Environment variables from `.env` can further override LLM settings.

### Key configuration parameters

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `llm.model` | `deepseek-chat` | — | LLM model name (DeepSeek V3) |
| `llm.temperature` | `0.7` | — | Sampling temperature |
| `llm.max_tokens` | `4096` | — | Max output tokens per call |
| `llm.timeout` | `120` | — | API timeout in seconds |
| `llm.max_retries` | `3` | — | Retry count on failure |
| `memory.framework` | `mem0` | — | Memory compressor: `mem0` / `amem` / `memorybank` |
| `benchmark.name` | `hotpotqa` | — | Benchmark: `hotpotqa` / `swebench` / `hotpotqa_hard` |
| `benchmark.num_samples` | `20` | — | Number of tasks to process |
| `skill_induction.variants` | `[traj_to_skill, memory_to_skill, hybrid_to_skill]` | — | Skill induction pathways to run |
| `evaluation.num_validation_runs` | `3` | — | Validation runs per skill |
| `output.experiment_dir` | `./experiments` | — | Output directory |
| `output.log_level` | `INFO` | — | Log level: `DEBUG` / `INFO` / `WARNING` / `ERROR` |

### Creating a custom config

Create a new YAML file in `configs/`, only specifying the parameters you want to override:

```yaml
# configs/my_experiment.yaml
llm:
  model: deepseek-chat
  temperature: 0.5

benchmark:
  name: swebench
  num_samples: 10

memory:
  framework: amem

skill_induction:
  variants:
    - hybrid_to_skill
```

Run it with:
```bash
python scripts/run_mvp.py --config my_experiment
```

---

## Pipeline Details

### 1. Trajectory Collection (`src/trajectory/collector.py`)

- Drives a **ReAct agent** (Thought → Action → Observation loop) through each benchmark task
- Records every step as a `TrajectoryStep` with type (`thought` / `action` / `observation` / `error`)
- Stops when the agent outputs `Answer:` or hits `max_steps` (default: 50)
- Each trajectory is saved as a JSON file

### 2. Memory Compression (`src/memory/compressor.py`)

Three strategies available (set via `memory.framework`):

| Framework | Strategy | Key Feature |
|-----------|----------|-------------|
| `mem0` | Flat key-value extraction | Simple, production-style, each entry is independent |
| `amem` | Agentic self-organising (2-pass) | Pass 1: atomic extraction → Pass 2: reflection + linking + merging |
| `memorybank` | Hierarchical tiering | Entries scored by importance → core/working/ephemeral tiers, ephemeral entries are forgotten |

All compressors output a `MemoryStore` containing `MemoryEntry` objects with `content`, `category` (fact/rule/procedure/insight), `specificity_score`, and `importance`.

### 3. Skill Induction (`src/skill_induction/`)

Three competing pathways:

| Variant | Input | Approach |
|---------|-------|----------|
| `traj_to_skill` | Trajectory only | Direct extraction from raw interaction steps |
| `memory_to_skill` | Memory only | Extraction from compressed, structured memory |
| `hybrid_to_skill` | Trajectory + Memory | Combines both sources with evidence retrieval |

Each produces a `Skill` object with: `name`, `description`, `preconditions`, `procedure` (steps), `constraints`, `facts`, `rules`.

### 4. Evaluation (`src/evaluation/evaluator.py`)

For each skill:
- **Success rate:** Re-runs validation tasks with the skill injected as a system prompt, checks if the answer is correct
- **Compression ratio:** `tokens(trajectory) / tokens(skill)` — higher = more compact
- **Variant comparison:** Aggregates metrics across all tasks and ranks the 3 pathways

---

## Benchmarks

### Supported benchmarks

| Name | Dataset | Task Type | Source |
|------|---------|-----------|--------|
| `hotpotqa` | HotpotQA (distractor, validation) | Multi-hop reasoning QA | `hotpotqa/hotpot_qa` |
| `hotpotqa_hard` | HotpotQA hard subset | Hard multi-hop QA | `hotpotqa/hotpot_qa` (filtered) |
| `swebench` | SWE-bench Lite | Code bug-fixing | `princeton-nlp/SWE-bench_Lite` |

### First-run dataset download

On the first run, HuggingFace `datasets` library will download and cache the dataset. This is automatic and only happens once:
- HotpotQA: ~200MB
- SWE-bench Lite: ~50MB
- Cache location: `~/.cache/huggingface/datasets/`

---

## Testing

### Run all tests

```bash
python -m pytest tests/ -v
```

### Run specific test files

```bash
# Config loading tests (no network required)
python -m pytest tests/test_config.py -v

# Data model tests (no network required)
python -m pytest tests/test_models.py -v

# Integration tests (requires network for HuggingFace dataset download)
python -m pytest tests/test_integration.py -v
```

### Test categories

| Test File | Network Required | LLM Required | What It Tests |
|-----------|-----------------|--------------|---------------|
| `test_config.py` | No | No | Config loading, deep merge, project root |
| `test_models.py` | No | No | Pydantic model validation, serialisation, computed properties |
| `test_integration.py` | Yes (HuggingFace) | No | Benchmark loader (real data), compressor factory, tiering logic |

---

## Output Structure

After running an experiment, the output directory looks like:

```
experiments/mvp_locomo/
├── trajectories/
│   ├── hotpotqa_xxx.json          # Raw trajectory for each task
│   └── ...
├── memories/
│   ├── hotpotqa_xxx.json          # Compressed memory for each task
│   └── ...
├── skills/
│   ├── traj_to_skill/
│   │   ├── hotpotqa_xxx.json      # Skill from variant 1
│   │   └── ...
│   ├── memory_to_skill/
│   │   ├── hotpotqa_xxx.json      # Skill from variant 2
│   │   └── ...
│   └── hybrid_to_skill/
│       ├── hotpotqa_xxx.json      # Skill from variant 3
│       └── ...
├── results/
│   ├── traj_to_skill.jsonl        # Evaluation results for variant 1
│   ├── memory_to_skill.jsonl      # Evaluation results for variant 2
│   └── hybrid_to_skill.jsonl      # Evaluation results for variant 3
├── comparison.json                # Final variant comparison summary
└── logs/
    └── skillforge_YYYY-MM-DD.log  # Detailed execution log
```

### Key output files

**`comparison.json`** — The final result comparing all 3 variants:
```json
{
  "traj_to_skill": {
    "num_skills": 20,
    "avg_success_rate": 0.65,
    "avg_compression_ratio": 8.5
  },
  "memory_to_skill": {
    "num_skills": 20,
    "avg_success_rate": 0.70,
    "avg_compression_ratio": 12.3
  },
  "hybrid_to_skill": {
    "num_skills": 20,
    "avg_success_rate": 0.75,
    "avg_compression_ratio": 10.1
  }
}
```

**Individual skill JSON** — Each skill file contains:
```json
{
  "skill_id": "uuid",
  "name": "Multi-hop QA Reasoning",
  "description": "...",
  "preconditions": ["..."],
  "procedure": ["Step 1: ...", "Step 2: ..."],
  "constraints": ["..."],
  "facts": ["..."],
  "rules": ["..."],
  "source_variant": "hybrid_to_skill"
}
```

---

## Troubleshooting

### Common issues

| Problem | Cause | Solution |
|---------|-------|----------|
| `DEEPSEEK_API_KEY is not set` | Missing `.env` file or empty key | Run `cp .env.example .env` and fill in your API key |
| `ModuleNotFoundError: No module named 'xxx'` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `Connection error` / `timeout` on LLM calls | Network issue or API overload | Check internet; increase `llm.timeout` in config; the client auto-retries 3 times with exponential backoff |
| `Unsupported benchmark: xxx` | Invalid benchmark name | Use one of: `hotpotqa`, `swebench`, `hotpotqa_hard` |
| `Unsupported memory framework: xxx` | Invalid framework name | Use one of: `mem0`, `amem`, `memorybank` |
| HuggingFace dataset download fails | Network / proxy issue | Set `HF_ENDPOINT=https://hf-mirror.com` for mirror; or manually download to `~/.cache/huggingface/datasets/` |
| Experiment takes too long | Too many samples | Reduce `benchmark.num_samples` (e.g. set to 5 for a quick test) |
| `JSONDecodeError` in memory compression | LLM returned non-JSON | The compressor has a fallback: it wraps the raw response as a single memory entry and continues |

### Quick smoke test (minimal run)

To verify everything works with minimal cost/time:

```bash
# Create a minimal config
cat > configs/smoke_test.yaml << 'EOF'
llm:
  model: deepseek-chat
  temperature: 0.5

benchmark:
  name: hotpotqa
  num_samples: 2

memory:
  framework: mem0

skill_induction:
  variants:
    - traj_to_skill

evaluation:
  num_validation_runs: 1
EOF

# Run smoke test
python scripts/run_mvp.py --config smoke_test
```

This processes only 2 tasks with 1 variant, completing in ~2-3 minutes.

### Using a different LLM provider

SkillForge uses the OpenAI-compatible API format. To use a different provider, update `.env`:

```bash
# Example: using a local Ollama instance
DEEPSEEK_API_KEY=ollama
DEEPSEEK_BASE_URL=http://localhost:11434/v1
DEEPSEEK_MODEL=llama3

# Example: using OpenAI directly
DEEPSEEK_API_KEY=sk-your-openai-key
DEEPSEEK_BASE_URL=https://api.openai.com/v1
DEEPSEEK_MODEL=gpt-4o
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
