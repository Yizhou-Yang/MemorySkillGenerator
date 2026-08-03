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
