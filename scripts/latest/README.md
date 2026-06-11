# V6 Experiment Runner

Single experiment runner for the SkillForge V6 ablation (A/B/C groups)
on GAIA / ALFWorld / LoCoMo.

## File

| File | Description |
|------|-------------|
| `latest_runner.py` | Sequential iterative training + cross-agent skill quality critic. Metrics: EM (GAIA, LoCoMo) and pass@1 (ALFWorld), aligned with competing papers (Voyager, Reflexion, SkillWeaver, Mem0). |

## Ablation Groups

- **A (Baseline)** — original prompt, no augmentation.
- **B (Raw)** — inject experiences without AI refinement.
- **C (AI-Refined)** — inject AI-refined experiences that passed the cross-agent
  quality critic (`quality >= QUALITY_THRESHOLD = 5`).

## Design Choices

1. **No oracle-driven retry on QA tasks.** In real deployments we cannot tell
   whether a GAIA / LoCoMo answer is correct. Instead, every candidate
   experience is rated 0–10 by an independent LLM critic; only experiences
   above the threshold enter the library.
2. **ALFWorld retry kept** because it has a real `won` signal from the
   environment.
3. **Metrics** match competing literature: Exact Match (string-normalized
   equality with substring relaxation) for QA tasks, pass@1 for ALFWorld.

## Run

```bash
python scripts/v6/latest_runner.py
```

Results land in `experiments_results/latest/`.
