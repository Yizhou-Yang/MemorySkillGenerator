# Archived experiment data — NOT usable for the paper

Kept for reference only. Nothing here may be reported, pooled, or cited. The
analysis scripts do not read this directory (they read
`experiments_results/$BASE/<model>/<bench>/trace.jsonl`, `BASE=latest_evolving`),
which is the point of moving it out of the live tree rather than leaving it there.

## hy3/ and hy3-preview-ioa/ (5721 rows) — rerun pending

`hy3-preview-ioa` and `hy3` are the **same model**. Every one of
`hy3-preview-ioa`'s rows (600 gaia / 1200 gaia2 / 600 locomo) is byte-identical
to a row inside `hy3`'s trace, tagged `code_rev=6bc02f9f`; `hy3`'s own newer rows
are the `ac6fb30f` ones (331 / 290 / 300).

Three reasons this data cannot back a claim, none of them fixable in
post-processing:

1. **Mixed `code_rev` within a single arm.** `hy3`'s `no_mem` holds both
   `6bc02f9f` and `ac6fb30f` rows, so half the arm ran different code. Rev
   uniformity is required *per arm* (`v2_gate.py` G4).
2. **`6bc02f9f` is unauditable.** The commit does not exist — not in the repo,
   not in the reflog; it predates the 2026-07-02 force-squash of `origin/main`.
   There is no way to answer "what code produced these numbers", which is the
   exact question `v2_gate.py` asks when it says "audit which rows predate the
   fix". `ac6fb30f` survives and is an *ancestor* of `69a672c8`, the rev behind
   the trusted deepseek-v4-pro set.
3. **No C arm at all**, so a rerun was required regardless.

There is no salvageable subset: keeping only `ac6fb30f` leaves a complete A arm
but a 31-row B stub (0 finished), and keeping only `6bc02f9f` gives A+B at a rev
that no longer exists. hy3 is being rerun A/B/C from scratch on the current rev.

Do not "fix" this by merging revs or by picking the better-looking one.

## Deleted rather than archived (recoverable via git history)

* `latest_evolving/deepseek-v3.2/` (482 rows) — chain died, no iter2 despite
  `iter_total=3`. Its former Table 2 cells came from aborted partial iterations
  (gaia2 n=50, locomo n=32) printed beside n=100 numbers with no n annotation.
* `latest_evolving/llama-33/gaia2/` (600 rows) — every row scored exactly 0.0;
  460/511 answered rows were raw `<|python_tag|>` text. vLLM tool-parser
  mismatch, fixed in b1efd65d. A dead harness, not a zero baseline.
* `latest_evolving/llama-33` curated_patch rows (114) — iter0 only,
  `patch_injected=0`, never finished: an aborted arm, not a result.
