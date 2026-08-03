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

Table 2 does not come from a trace; the native-protocol harness writes its own
results, and each answerer is its own run:

- answerer HY3, all arms but A-Mem: `locomo_native/hy3/SUMMARY.txt`
- answerer HY3, A-Mem: `locomo_native/hy3_amem/results.jsonl`. The `8.0` in
  `hy3/SUMMARY.txt` is the pass that hit the A-Mem recall bug; this is its rerun.
- answerer GPT-5.5, all five: `locomo_native/FINAL_COMPARISON.txt`

`locomo_native/main/FINAL_SUMMARY.txt` is **superseded** and reports `ours 49.5`,
below both baselines. Its own `CATEGORY_BREAKDOWN.txt` says why: that run used
extraction from before `cab3c334`, and its `nomem` arm was a muzzled-prompt
artifact scoring ~0 by construction. It is kept because deleting a superseded run
is how a rerun gets mistaken for a contradiction — read the header before
quoting any number out of that directory.

## The rule this exists to enforce

A result is not data until it is on a branch. Traces that live only on a leased
box are one reclamation away from unbacked cells, and a box gives no warning.
Commit and push each arm as it lands, not at the end of a sweep.
