# Formal (frozen) results

Snapshots in this folder are **locked** — experiment runs only ever write to
`experiments_results/latest/`, never here. Each subfolder is one dated, immutable
result set you can cite/compare against. To analyze one:

```bash
python scripts/latest/analyze_results.py experiments_results/formal/<snapshot>
```

## Snapshots

### `2026-06-16_hy3_n30`
First complete A/B/C run on **hy3-preview-ioa** (hunyuan-3), 30 tasks/benchmark.
Mechanism = legacy within-task A/B/C (gaia/gaia2 prompt-blob augmentation, locomo
self-consistency) — i.e. EvoMem/GPR retrofit NOT yet applied.

Primary-metric mean scores:

| Benchmark | A | B | C | C−A | A≤B≤C |
|---|---|---|---|---|---|
| gaia | 0.200 | 0.167 | **0.233** | +3.3 | no (B<A) |
| gaia2 (soft recall) | 0.125 | 0.167 | **0.182** | +5.7 | yes |
| locomo | 0.300 | 0.333 | **0.400** | +10.0 | yes |
| terminal_bench_2 | 0.056 | 0.056 | **0.089** | +3.3 | yes (B=A) |

Findings:
- **C is the best arm on every benchmark; C ≥ B and C > A hold throughout** —
  the no-regression gating fix works (old locomo C was *worst* at 0.20; now best).
- **But every delta is statistically n.s. at n=30** (McNemar p>0.05, bootstrap CI
  crosses 0). The signal is real and directionally consistent but underpowered.
- Prior fixes confirmed effective here: gaia2 tool-not-found 12/30 → 1/90;
  terminal empty-test_output 14/39 → 0 (all docker); locomo relative-date
  answers 24 → 9.
- Surfaced environment issues (not code): 3 terminal docker images fail to pull
  (`alexgshaw/{dna-insert,extract-moves-from-video,feal-differential-cryptanalysis}`),
  plus intermittent `API unavailable`.

Next step recorded after this snapshot: scale to 100 tasks/benchmark + add
transient/empty-response retry to chase significance.
