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

## Status as of 2026-08-03

27 ok, 1 failed, 8 missing. The gaps are all one event: **the gpu3 container was
reclaimed while nine cells' traces lived only on its disk.** The gateway now
answers `acl_denied`, so those files are not coming back.

| Paper cell | Source that is gone |
|---|---|
| Table 1, GAIA / GPT-5.5 / A-Mem — 23.00 and 28.99 | the `amem` arm of `gpt55full/gpt-5.5/gaia`; what is on `main` is a 20-task partial |
| Table 1, GAIA2 / HY3 / CuratorMem — 42.00 | `hy3g2fix/hy3/gaia2` — the record-fix rerun, never pushed |
| Table 1, τ² / HY3 — Patched, A-Mem, Mem0, CuratorMem | `hy3tau2/hy3/tau2` was pushed mid-sweep; only `no_mem` and one `raw_patch` iteration made it |
| Figure 2a, L=500 (24.80) and L=∞ (26.60) | `hy3dose500` is a 12-task partial, `hy3dose0` was never pushed |

Everything else — all of Table 1's GAIA rows, all of Table 2, the L=900 point —
reproduces from files in this checkout.

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
