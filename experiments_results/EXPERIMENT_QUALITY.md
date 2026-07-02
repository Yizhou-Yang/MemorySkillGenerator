# Experiment Quality Gates — how to vet a sweep before trusting any number

This is the checklist for deciding whether an experiment run is **trustworthy**.
Run it on every sweep *before* a single number is quoted, put in a table, or
compared across models. Most "results" that look interesting are actually one of
the failure modes at the bottom of this page. When in doubt, a run is **guilty
until proven innocent**.

You do not need any external document to use this guide — everything you need is
in this repo (`scripts/latest/breakdown.py`, `scripts/latest/analyze_results.py`,
and the `trace.jsonl` files themselves).

---

## 0. What the experiment is

Every benchmark is run under three **arms** (the `group` field in each trace row):

| Arm | Trace key¹ | What it does |
|-----|-----------|--------------|
| **A** | `A_baseline` | No memory. The plain agent. This is the control. |
| **B** | `B_evomem` | **Raw patch memory.** Injects the raw record of what worked on *earlier iterations of the same task* (chain-scoped), verbatim. |
| **C** | `C_gpr` | **Curated patch memory** (the method under test). Same patches as B, but each is refined (generalized + a causal lesson), scored by an independent critic, low-quality ones enriched rather than dropped, retrieval is effectiveness-weighted, and failed attempts contribute an "avoid this" channel. |

¹ Trace keys are **legacy code identifiers** and are on the rename list — read
`A_baseline`/`B_evomem`/`C_gpr` as simply **A/B/C**. (Some older frozen snapshots
use `B_evoarena`/`C_skillforge` for B/C — same meaning.)

**The claim being tested is C > B** (curation beats raw patches), with B > A as a
secondary check (memory beats no-memory). A run that cannot cleanly compare C
against B has told you nothing about the method, no matter how the means look.

### The one mental model that explains most failures

Patch memory is **feedback across iterations of the *same* task** (a version
chain), **not** cross-task transfer. Retrieval is **chain-scoped**: a patch is
only visible to later iterations of the same `task_id` (or, for LoCoMo, the same
conversation session).

Consequence: on a **single pass** over independent tasks (GAIA / GAIA2 / TB2 run
once each), there is no in-chain history, so **B and C inject nothing and collapse
onto A**. Any A/B/C spread you see there is pure sampling noise — *not* a result.
To actually exercise memory on those benchmarks you must run **iteration chains**:

```bash
ITER_CHAIN=3 bash scripts/latest/run_all_models.sh   # each task run 3×; memory threads across iterations
```

With `ITER_CHAIN=K`, iteration `0` has no prior history (injects nothing, by
design); injection can only happen on iterations `1..K-1`. LoCoMo is the exception
— it has real multi-session chains, so it exercises memory even at `ITER_CHAIN=1`.

---

## The gates (run in order; stop at the first red)

### Gate 0 — Completeness

All three arms must be present with the expected row count. The main table uses
the **final** iteration, so a missing or partial C arm means there is **no result
for the method**.

```bash
python scripts/latest/analyze_results.py experiments_results/latest/<model>
# look for "!! incomplete: missing groups [...]" — that benchmark is not done.
```

Expected `n`: with `ITER_CHAIN=3` and 100 tasks, each arm has ~300 rows
(100 tasks × 3 iterations). A run that lists only A and B, or an arm far below the
expected count, is **in progress or crashed** — finish it with `RESUME=1` before
reading anything into it.

### Gate 1 — Error rate (catch infra/dataset failures masquerading as results)

An arm scoring ~0 is almost never a real finding — it is usually the harness
failing to run. **Per-arm error rate must be ≈ 0.**

```bash
python - <<'PY'
import json, collections, glob, os
root = "experiments_results/latest"   # or a specific <model> dir
for f in sorted(glob.glob(f"{root}/**/trace.jsonl", recursive=True)):
    by = collections.defaultdict(lambda: [0, 0])
    for line in open(f):
        line = line.strip()
        if not line: continue
        r = json.loads(line); g = r.get("group")
        by[g][0] += 1
        if r.get("error"): by[g][1] += 1
    bench = os.path.basename(os.path.dirname(f))
    for g in sorted(by):
        n, e = by[g]
        flag = "  <-- BROKEN" if n and e / n > 0.1 else ""
        print(f"{bench:20s} {g:12s} n={n:4d} error={e:4d} ({100*e//max(n,1)}%){flag}")
PY
```

If an arm shows a high error rate, open one of its rows: an `error` string plus
`prof_llm_calls: 0` and `response: ""` means the agent **never ran** (missing
dataset, container that would not pull, API outage). That arm is **invalid** —
fix the environment and re-run it; do not report its score.

### Gate 2 — Injection actually fired (the core integrity check)

If memory never got injected, then B and C are just re-samples of A and every
"improvement" is noise. `breakdown.py` reports this directly.

```bash
python scripts/latest/breakdown.py experiments_results/latest/<model>
```

In the **patch-injection isolation** table, for each benchmark that was run as an
iteration chain (or LoCoMo at any setting):

- **B and C must both show `injected n > 0`.** If either shows the
  `[!] NO patches were injected` warning on a chain run, that arm's memory did not
  fire — a bug (or a single-pass config). The arm is A-equivalent and any C-vs-B
  comparison on it is meaningless.
- **Both arms should inject at similar rates.** If B injects hundreds of times but
  C injects zero (or vice-versa), you have found an **arm-specific injection bug** —
  the run is not a valid A/B/C comparison. (This has happened: B silently stopped
  injecting on LoCoMo while C kept working, so a "significant C > B" was really
  just "C vs a broken baseline".)
- Ideally, **injected accuracy > not-injected accuracy** for C — that is the
  method helping. If injecting consistently *lowers* accuracy, the curation is
  hurting and that is a finding worth chasing, not hiding.

### Gate 3 — Significance (don't quote a mean without it)

Small EM gaps at small `n` are noise. Use the paired test, and prefer the paired
delta + confidence interval over raw group means.

```bash
python scripts/latest/analyze_results.py experiments_results/latest/<model>
```

- Read the **paired** `C vs B` / `B vs A` lines: `Δ`, the 95% CI, and the McNemar
  p-value. A CI that crosses 0 (or `n.s.`) is not a positive result.
- `n ≈ 30` is underpowered — a 1-task swing is ~3 pp and almost everything comes
  out `n.s.`. **Target `n ≈ 100`** per arm for usable intervals.
- The `ordering A<=B<=C` line is a quick sanity flag, not evidence on its own.

### Gate 4 — Absolute scores are plausible

If **all three** arms are implausibly low (e.g. every arm near 0 on a benchmark
peers solve at 30–50%), suspect a **scoring/parsing bug**, not a hard benchmark —
check the grader before the agent. (Past examples: a GAIA answer-extraction bug
that shadowed correct full responses; a TB2 pytest-output parser that scored
partial passes as 0.) Also confirm which metric a number is: `report.json` means
can include **soft recall** (fractional), while `analyze_results.py` uses binary
**EM** on the shared-task subset — the two can diverge, so never mix them in one
table.

---

## Known failure modes (seen in real runs)

| Symptom | Almost always means | Action |
|---|---|---|
| A = B = C, deltas tiny | Memory never injected — single-pass run, no `ITER_CHAIN`, or a chain-scoping bug | Re-run with `ITER_CHAIN=3`; confirm Gate 2 |
| One arm ≈ 0% / high error rate | Dataset or container missing; API outage — agent never ran | Fix env (persist datasets **off `/tmp`**, pre-pull images), re-run that arm |
| B injects but C doesn't (or vice-versa) | Arm-specific injection bug | Not a valid comparison — fix before reporting |
| "Significant C > B" but B injected 0× | B is a broken second baseline, not raw-patch memory | The C-vs-B claim is unproven; fix B, re-run |
| All arms implausibly low | Grader/parse bug, not task difficulty | Audit the scorer before the agent |
| Means quoted but every test `n.s.` | Underpowered (`n≈30`) | Scale to `n≈100`; report Δ + CI |

## Quick verdict rubric

- **Trust it** only if: all A/B/C complete (Gate 0) · error rate ≈ 0 (Gate 1) ·
  B *and* C injected on chain runs (Gate 2) · the C-vs-B delta is significant with
  a CI clear of 0 (Gate 3) · absolute scores are sane (Gate 4).
- **Re-run** if any arm is incomplete, broken, or non-injecting.
- **Discard** any arm whose score is an artifact of errors — never let it into a
  table "for completeness".

See [`README.md`](README.md) for the results layout and trace schema, and
[`../scripts/latest/EXPERIMENT_PLAN.md`](../scripts/latest/EXPERIMENT_PLAN.md) for
how a full sweep is launched.
