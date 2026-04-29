# SkillForge

> **Experience-Augmented Agent Framework with Version-Aware Skill Evolution**
>
> SkillForge extracts structured experiences from agent execution trajectories,
> maintains an EvoMem-style version history with patch tracking, and injects
> relevant skills into future tasks via semantic retrieval — improving agent
> performance across dynamic and static benchmarks.

---

## Key Contributions

1. **Dual-Feedback Experience Recording**: Learns from both successful and failed executions, tracking positive strategies and negative pitfalls simultaneously.
2. **EvoMem-Style Patch History**: Git-like version tracking for each skill — records what changed, why, and how scores evolved across attempts.
3. **Cross-Agent Critic with Forced Refinement**: Independent LLM evaluator scores skill quality; low-quality skills are *enriched* (never discarded), preserving all information.
4. **Task-Type-Aware Injection Routing**: Automatically classifies tasks (agentic / QA / embodied) and routes to appropriate injection format.
5. **Zero Information Loss**: The system never compresses, summarizes, or removes information from skills — only adds context, failure modes, and recovery strategies.

---

## Architecture

```
                        ┌──────────────────────────┐
                        │     Agent Execution       │
                        │   (LLM agent loop)        │
                        └────────────┬─────────────┘
                                     │ trajectory + score
                                     ▼
              ┌──────────────────────────────────────────┐
              │           analysis.py                     │
              │  Trajectory → Experience extraction       │
              │  • Format-adaptive action key extraction  │
              │  • Fuzzy matching vs oracle (rapidfuzz)   │
              │  • 4-category failure classification      │
              └──────────────────┬───────────────────────┘
                                 │ Experience
                                 ▼
              ┌──────────────────────────────────────────┐
              │           refine.py                       │
              │  Version-Conditioned AI Refinement        │
              │  • LLM generalizes with placeholders      │
              │  • Extracts causal lesson (why it worked/ │
              │    failed)                                │
              │  • Analyzes patch_history diff chain      │
              │  • Cross-agent quality evaluation         │
              │  • Critic-driven enrichment (never        │
              │    discards — only expands)               │
              └──────────────────┬───────────────────────┘
                                 │ refined experience
                                 ▼
              ┌──────────────────────────────────────────┐
              │         experience.py                     │
              │  ExperienceLibrary                        │
              │  • EvoMem-style version tracking          │
              │  • Semantic retrieval (sentence-          │
              │    transformers + sklearn TF-IDF cosine   │
              │    fallback)                              │
              │  • Per-experience effectiveness weighting │
              │  • Patch history (append-only log)        │
              └──────────────────┬───────────────────────┘
                                 │ top-k relevant experiences
                                 ▼
              ┌──────────────────────────────────────────┐
              │          injection.py                     │
              │  Full-Context Prompt Injection            │
              │  • gate.py: classify_task_type            │
              │    (agentic / qa / embodied)              │
              │  • Route: qa → enhanced hints with        │
              │    pitfall warnings; agentic → full       │
              │    experience injection                   │
              │  • Zero information loss — injects all    │
              │    relevant experience without truncation │
              └──────────────────┬───────────────────────┘
                                 │ augmented prompt
                                 ▼
              ┌──────────────────────────────────────────┐
              │          gate.py                          │
              │  Applicability Gate                       │
              │  • Task complexity assessment             │
              │    (simple / moderate / complex)          │
              │  • Task type classification via           │
              │    structural signals (no hardcoded       │
              │    keyword lists)                         │
              │  • Augmentation decision                  │
              └──────────────────┬───────────────────────┘
                                 │
                                 ▼
                        ┌──────────────┐
                        │  Next Task   │  (iterative feedback loop)
                        └──────────────┘
```

---

## Core Modules

| Module | Responsibility | Key Dependencies |
|--------|---------------|-----------------|
| `experience.py` | Experience dataclass with EvoMem-style version tracking; ExperienceLibrary with semantic embedding retrieval and per-experience effectiveness weighting | sentence-transformers, sklearn |
| `analysis.py` | Format-adaptive trajectory analysis; greedy ordered action matching via fuzzy string similarity; 4-category failure classification (tool_failure / over_action / task_mismatch / model_failure) | rapidfuzz |
| `gate.py` | Task type classification (agentic / qa / embodied) using structural signals; task complexity assessment; augmentation gating | — |
| `injection.py` | Task-type-aware routing; formats success experiences (with evolution context) and failure experiences (with patch history + recovery strategies); full-context injection without truncation | — |
| `refine.py` | Version-conditioned AI refinement; cross-agent quality evaluation (independent LLM judge); critic-driven enrichment for low-quality skills (adds failure modes, recovery strategies, preconditions — never removes information) | json_repair |

---

## Design Principles

### 1. Zero Information Loss

The system **never compresses, summarizes, or discards** skill content. When a cross-agent critic scores a skill below threshold, the response is forced refinement (enrichment) — adding context, failure modes, and recovery strategies on top of existing content. Content is only replaced if the new version is strictly longer.

### 2. EvoMem-Style Version Tracking

Each experience maintains an append-only `patch_history` recording:
- Score deltas across attempts (`score_delta`)
- Outcome transitions (`failure → partial → success`)
- Steps fixed and steps still missing
- New steps added and old steps removed

This enables the system to learn *how* a skill evolved, not just its final state.

### 3. Dual-Feedback Learning

Unlike systems that only learn from success (or only from failure), SkillForge records both:
- **Success experiences**: Highlight what worked — tool chains, strategies, causal reasoning
- **Failure experiences**: Highlight pitfalls — missing steps, root causes, avoidance notes

Both are injected at retrieval time, giving the agent positive guidance and negative warnings simultaneously.

### 4. Cross-Agent Critic Evaluation

An independent LLM evaluator scores each skill on 4 dimensions:
- **Actionability** (0–3): Are steps concrete and reproducible?
- **Generalizability** (0–3): Would this help on different but similar tasks?
- **Correctness** (0–2): Is the approach logically sound?
- **Novelty** (0–2): Does it provide non-obvious insight?

Skills scoring below threshold undergo forced enrichment (never rejection).

### 5. Effectiveness-Weighted Retrieval

Each experience tracks its historical injection effectiveness (score delta when used). Experiences that historically hurt performance are downweighted (min 0.3×); those that helped are upweighted (max 1.5×). This creates a self-correcting retrieval signal.

---

## Experiment Protocol

The evaluation follows an **iterative train-then-test** protocol:

1. **Split**: Each benchmark is split into train (50%) and test (50%) sets
2. **Train phase**: Tasks are executed sequentially; after each task, the experience is recorded, refined, and added to the skill library (the library grows incrementally)
3. **Test phase**: The accumulated skill library is frozen; test tasks are executed with skill injection enabled
4. **Metrics**: Task-level accuracy (EM / soft recall / pass@1) on the held-out test set

### Benchmarks

| Benchmark | Domain | Tasks | Metric |
|-----------|--------|-------|--------|
| GAIA (HuggingFace) | Multi-step QA | 50 | Exact Match |
| ALFWorld | Embodied reasoning | 40 | Pass@1 (binary won) |
| LoCoMo | Long conversation memory | 50 | Exact Match |
| GAIA2 | Agentic tool-use (CLI) | 50 | Soft Recall (action sequence) |
| SWE-bench (Dynamic) | Software engineering | 30 | Patch correctness (LLM judge) |

---

## Project Structure

```
SkillForge/
├── src/
│   └── v6/
│       ├── __init__.py         # SkillForgeV6 orchestrator (record → version → refine → inject)
│       ├── experience.py       # Experience dataclass + ExperienceLibrary + semantic retrieval
│       ├── analysis.py         # Trajectory analysis + failure classification
│       ├── refine.py           # AI refinement + cross-agent evaluation + critic enrichment
│       ├── injection.py        # Task-type-aware prompt injection (full context)
│       └── gate.py             # Task type classification + complexity assessment
├── benchmarks/
│   └── loader.py               # Unified loader for all 5 benchmarks
├── scripts/
│   └── v6/
│       └── latest_runner.py    # Main experiment runner (train/test protocol)
├── configs/                    # YAML experiment configurations
├── tests/                      # Unit and integration tests
├── pyproject.toml
└── requirements.txt
```

---

## External Dependencies

All algorithmic components use established libraries — no hand-written similarity or NLP algorithms:

| Library | Usage |
|---------|-------|
| `sentence-transformers` | Semantic embedding for experience retrieval (all-MiniLM-L6-v2) |
| `sklearn` | TF-IDF vectorization + cosine similarity (fallback retrieval) |
| `rapidfuzz` | Fuzzy string matching for action sequence alignment |
| `json_repair` | Robust JSON extraction from LLM responses |

---

## Quick Start

```bash
pip install -r requirements.txt

# Run the full 5-benchmark experiment
python scripts/v6/latest_runner.py
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
