# Where every number in the paper comes from

`verify.py` recomputes each one from the file named beside it. Run it from the
repo root before any submission:

```
python3 experiments_results/_paper/verify.py
```

`ok` means the file still reproduces the printed number. `FAIL` means the number
moved. `MISSING` means the source is gone and that cell is currently unbacked.

A cell is: the arm's mean over the tasks it completed at the final iteration
(2), replicates averaged first — τ² runs each task twice. Δ in Table 1 is the
plain difference of two printed cells, so the row is self-checking.

## Status

**36 ok, 0 failed, 0 missing.**

Cells are read from `_paper/cells/*.jsonl` — each full trace distilled to the
columns a mean is computed from (`task_id`, `group`, `iteration`, `score`, `em`,
`error`). The full traces carry every prompt and response and are far too large
to push; `hy3g2fix`'s alone is 143 MB. Regenerate the distilled copies with
`mkcells.py` after a rerun. The originals live on `any4` under
`/data/workspace/{MemorySkillGenerator,MSG2}/experiments_results/`.

Two of the eight files came from the `MSG2` checkout rather than this one, which
is worth knowing before hunting for a number: the two checkouts use the same
`RESULTS_BASE` names for different content, and this checkout's `hy3dose500`
stops at 12 tasks while `MSG2`'s has all 100.

The per-task-type breakdown does not come from a trace; the native-protocol
harness writes its own results, and each answerer is its own run:

| Breakdown column | File under `locomo_native/` |
|---|---|
| answerer HY3 — Raw, Patched, Mem0, CuratorMem | `hy3/` (`SUMMARY.txt`, `results.jsonl`) |
| answerer HY3 — A-Mem | `hy3_amem/` — the rerun. The `8.0` in `hy3/SUMMARY.txt` is the pass that hit the A-Mem recall bug |
| answerer GPT-5.5 — Raw, A-Mem, Mem0 | `staged/` |
| answerer GPT-5.5 — Patched | `raw_arm/` |
| answerer GPT-5.5 — CuratorMem | `ours_sdk/` |
| both answerers, assembled | `FINAL_COMPARISON.txt` |

Archived off main to `data/locomo-iteration-chain-archive-2026-08-03`: the
iteration-chain LoCoMo runs (superseded by the main table's LoCoMo rows), plus
`locomo_native/main/`, `gpt55_grok/`, `ours_sdk_prev/` and `mini/`.

`locomo_native/main/` is the one worth naming. It reports `ours 49.5`, below both
baselines, and reads as a flat contradiction of the reported numbers until you
read its own `CATEGORY_BREAKDOWN.txt` header: pre-`cab3c334` extraction, and a
`nomem` arm that was a muzzled-prompt artifact scoring ~0 by construction. It is
on the archive branch rather than deleted for exactly that reason.

Also worth knowing: in that harness `full` is the full-transcript context and
`raw` is the patch-memory arm. `full` is **not** the paper's Patched column.

## Sub-3B backbones — Table 1's Llama-3.2-3B rows, and Appendix A

`locomo_weak/{llama-3.2-1b,llama-3.2-3b,qwen2.5-1.5b}/`, run 2026-08-05 on the
V100 box, all five arms, `n=497` per cell. Answerers are served locally by vLLM in
`float16` (sm_70 has no `bfloat16`); only the grader is remote.

| File | What it is |
|---|---|
| `results.jsonl` | one row per (system, conversation, question) with `pred`, `gold`, `J`, `F1` |
| `results_judge55.jsonl` | the same rows for `ours`/`amem`/`nomem`, re-graded by `gpt-5.5` |

Two graders touched this block. `gpt-5.6-sol` graded `ours`, `amem` and `nomem`;
it then began returning `SERVICE_UNAVAILABLE` on ~90% of requests even when idle,
so `mem0` and `raw` were graded by `gpt-5.5`. Rather than compare arms across
graders, the three sol-graded arms were re-graded from their stored predictions —
no answer was regenerated, only the grader changed. The two agree to within 3.6
points and on every ordering. **`results_judge55.jsonl` is the one to read**;
`verify.py` substitutes it wherever it exists and falls back to `results.jsonl`
for `mem0`/`raw`, which were never sol-graded.

Worth knowing before quoting these: Mem0 beats CuratorMem on the judge metric on
all three backbones while CuratorMem beats Mem0 on token-F1 on all three. That is
checked, not a broken arm — identical question sets, zero empty predictions, zero
Mem0 ingestion skips. At 1B it is partly length (Mem0 averages 16.4 words to our
12.6 against a 4.9-word gold); at Qwen2.5-1.5B the lengths match and Mem0 is
simply judged right more often.

## GAIA2 under GPT-5.5 — Table 1's second GAIA2 row

| Column | Source |
|---|---|
| Raw, Patched, A-Mem, Mem0 | `gpt55full/gpt-5.5/gaia2` |
| CuratorMem | `gpt55g2act2/gpt-5.5/gaia2` |

The curated arm is a separate run because the earlier one is not the same
mechanism. Arm C used to serve only 19% of chains, which held the cell at
`33.73`; the endorsement key plus the measured dead-chain rule took coverage to
92% and the cell to `41.99`. Read-time budget 1500, `ITER_CHAIN=3`, critic and
judge `gpt-5.6-terra` (`gpt-5.6-sol` was returning 500s by then).

`no_mem` and `raw_patch` carry n=99, not 100 — one task errored out of those two
arms and is dropped rather than scored zero, which is the same rule used
everywhere else in the table.
