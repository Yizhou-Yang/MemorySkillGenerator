#!/usr/bin/env python3
"""v2-method acceptance gate — MUST pass on every rerun before its numbers are
trusted. Checks, per QA benchmark trace, that the frozen v2.2 read path is
actually live in the data (not just in the code):

  G1  C injection coverage at iter>=1 is ~100% (chain-index + fallback fixed
      the 63-72% silence; a low rate means the old gate behaviour is back)
  G2  C dose: mean/max rendered block within the budget (aug_len <= L + small
      prompt-wrapper slack), and C mean dose <= B mean dose on this benchmark
  G3  new-marker presence: "## Curated prior attempts" header, and at least
      one "Answer given then" (false-failure answer preservation) where the
      avoidance channel fired; "Actions used:" expected on gaia2
  G4  per-arm final-iteration completeness (n tasks with a row at
      iter_total-1) and ONE code_rev per trace

Usage: python scripts/latest/v2_gate.py <model> [BASE=latest_evolving]
Exit code 1 if any hard gate (G1/G2/G4) fails.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BUDGET = int(os.environ.get("C_INJECT_BUDGET_CH", "900"))
SLACK = 130          # headers + section titles around the budgeted blocks
COVERAGE_FLOOR = 0.95
# A trace with no C arm cannot be gated (see G1 below). Only turn this off for a
# sweep that is deliberately A/B-only.
REQUIRE_C = os.environ.get("REQUIRE_C", "1") == "1"


_LEGACY_MAP = {"A_baseline": "no_mem", "B_evomem": "raw_patch", "C_gpr": "curated_patch"}


def rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            r["group"] = _LEGACY_MAP.get(r.get("group", ""), r.get("group", ""))
            out.append(r)
    return out


def main() -> None:
    model = (sys.argv[1] if len(sys.argv) > 1 else "hy3").lower()
    base = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("BASE", "latest_evolving")
    hard_fail = False
    gated_any = False
    for bench in ["gaia", "gaia2", "locomo"]:
        rs = rows(PROJECT_ROOT / "experiments_results" / base / model / bench / "trace.jsonl")
        if not rs:
            print(f"[{bench}] no trace — skipped")
            continue
        gated_any = True
        # ── G0: infra health. Error rows (endpoint down / timeout) and
        # empty-response zero rows (agent loop never engaged) are not task
        # results; a run dominated by them is broken hardware, not a baseline
        # (llama-33: gaia 770 APIUnavailable rows, gaia2 433 empty rows). They
        # are excluded from the gate stats below, and a rate >20% hard-fails.
        _is_err = lambda r: bool(str(r.get("error") or "").strip()) \
            and not str(r.get("response") or "").strip()
        _is_empty = lambda r: not str(r.get("response") or "").strip() \
            and not str(r.get("error") or "").strip() \
            and not float(r.get("score") or 0.0)
        n_err = sum(1 for r in rs if _is_err(r))
        n_empty = sum(1 for r in rs if _is_empty(r))
        infra_rate = (n_err + n_empty) / len(rs)
        if n_err or n_empty:
            print(f"[{bench}] G0 infra: {n_err} error-rows + {n_empty} "
                  f"empty-response zero-rows = {100*infra_rate:.0f}% of {len(rs)}")
        if infra_rate > 0.20:
            print(f"  G0 ✗ infra-row rate {100*infra_rate:.0f}% > 20% — the "
                  "endpoint/harness was broken during this run; rerun it")
            hard_fail = True
        rs = [r for r in rs if not _is_err(r) and not _is_empty(r)]
        if not rs:
            print(f"[{bench}] all rows were infra failures — nothing to gate")
            continue
        # G0b dead benchmark: rows that ANSWERED but every one scored 0. Not a
        # baseline — a broken scorer or a harness that never executed the task.
        # The infra-rate check above cannot see this (these rows carry a real
        # response). llama-33 gaia2: 511 answered rows, all 0.0, because the
        # model's <|python_tag|> tool calls leaked into content unparsed.
        if not any(float(r.get("score") or 0.0) for r in rs):
            print(f"  G0 ✗ every one of {len(rs)} answered rows scored 0.0 — "
                  "the benchmark is dead (scorer or tool path broken), not a "
                  "zero baseline; fix it before reading any delta here")
            hard_fail = True
        kf = max(int(r.get("iter_total", 1) or 1) for r in rs) - 1
        # Rev uniformity is PER GROUP: keeping valid A/B rows from an earlier
        # rev while C reruns on the frozen rev is the sanctioned plan; only a
        # rev mix WITHIN one arm means half an arm ran different code.
        by_rev = {}
        for r in rs:
            by_rev.setdefault(r.get("group", "?"), set()).add(
                str(r.get("code_rev", ""))[:8])
        print(f"\n[{bench}] rev per group="
              f"{ {g: sorted(v) for g, v in sorted(by_rev.items())} } final_iter={kf}")
        for g, v in sorted(by_rev.items()):
            if len(v) > 1:
                print(f"  G4 ✗ arm {g} has MIXED code_rev {sorted(v)} — "
                      "audit which rows predate the fix")
                hard_fail = True
        doses = {}
        for g in ["no_mem", "raw_patch", "curated_patch"]:
            grs = [r for r in rs if r.get("group") == g]
            fin = {r["task_id"] for r in grs if (r.get("iteration", 0) or 0) == kf}
            late = [r for r in grs if (r.get("iteration", 0) or 0) >= 1]
            inj = [r for r in late if r.get("patch_injected")]
            lens = [r.get("aug_len") or 0 for r in inj]
            doses[g] = (sum(lens) / len(lens)) if lens else 0.0
            cov = (len(inj) / len(late)) if late else 0.0
            print(f"  {g}: finished n={len(fin)}  inj@iter>=1 {len(inj)}/{len(late)}"
                  f" ({100*cov:.0f}%)  dose mean={doses[g]:.0f} max={max(lens) if lens else 0}")
            # G4 arm completeness: an arm with rows but NONE at the final
            # iteration started and died mid-sweep; its rows are a partial run,
            # not a result. Every C gate below is conditional on C having late
            # rows, so without this an aborted C arm silently skips G1/G2/G3
            # and the trace reports PASS with the method never having run
            # (llama-33: C had 16 iter0 rows on gaia, 0 finished, PASS).
            if grs and not fin:
                print(f"  G4 ✗ arm {g}: {len(grs)} rows but NONE finished at "
                      f"iter {kf} — the arm started and died; partial run")
                hard_fail = True
            if not grs:
                print(f"  G4 ! arm {g}: absent from this trace")
            if g == "curated_patch" and late:
                if cov < COVERAGE_FLOOR:
                    print(f"  G1 ✗ C coverage {100*cov:.0f}% < {100*COVERAGE_FLOOR:.0f}%"
                          " — chain-index/fallback not effective")
                    hard_fail = True
                if lens and max(lens) > BUDGET + SLACK:
                    print(f"  G2 ✗ C max dose {max(lens)} > {BUDGET}+{SLACK}")
                    hard_fail = True
        # This gate exists to prove the C read path is live IN THE DATA. With no
        # C rows at iter>=1 there is nothing to prove, and a PASS here would be
        # vacuous — the most dangerous output this script can produce, since it
        # is the last thing between a broken sweep and the paper. Set
        # REQUIRE_C=0 for a deliberate A/B-only baseline sweep.
        if REQUIRE_C and not [r for r in rs if r.get("group") == "curated_patch"
                              and (r.get("iteration", 0) or 0) >= 1]:
            print("  G1 ✗ curated_patch has no rows at iter>=1 — the method arm "
                  "never ran, so C coverage/dose/markers are unevaluable. This "
                  "is NOT a pass (set REQUIRE_C=0 for an A/B-only sweep).")
            hard_fail = True
        if doses.get("curated_patch") and doses.get("raw_patch") and \
                doses["curated_patch"] > doses["raw_patch"] + 1:
            # The paper's below-B-dose claim is scoped to the AGENTIC
            # benchmarks (gaia2/TB2), where dose harm was measured. On
            # QA benches B's natural render can sit far below L (locomo
            # B≈225ch) and C legitimately exceeds it under the same cap.
            if bench == "gaia2":
                print(f"  G2 ✗ C mean dose {doses['curated_patch']:.0f} > B "
                      f"{doses['raw_patch']:.0f} (agentic below-B claim broken)")
                hard_fail = True
            else:
                print(f"  G2 note: C mean dose {doses['curated_patch']:.0f} > B "
                      f"{doses['raw_patch']:.0f} (allowed off-agentic; cap is L)")
        # G3 markers (soft): rendered-channel fingerprints in C prompts
        caugs = [r.get("augmented_prompt") or "" for r in rs
                 if r.get("group") == "curated_patch" and r.get("patch_injected")]
        if caugs:
            hdr = sum("## Curated prior attempts" in a for a in caugs)
            ans = sum("Answer given then" in a for a in caugs)
            raw = sum("As recorded" in a for a in caugs)
            act = sum("Actions used:" in a for a in caugs)
            print(f"  G3 markers: header {hdr}/{len(caugs)}, answer-lines {ans}, "
                  f"as-recorded {raw}, action-lines {act}")
            if hdr == 0:
                print("  G3 ✗ no v2.2 header in any C block — old code ran?")
                hard_fail = True
            # v2.3/2.4 scoping: gaia2/TB2 render action payload; answer-scored
            # benches render the raw attempt VERBATIM (superset of B) and must
            # carry no action lines (they displaced prose and drove gaia C−B
            # to −9.1pp).
            if bench == "gaia2" and act == 0:
                print("  G3 ✗ gaia2 has no action-lines — v2.3 payload missing")
                hard_fail = True
            if bench in ("gaia", "locomo"):
                if act > 0:
                    print(f"  G3 ✗ {bench} has {act} action-lines — v2.3+ "
                          "scoping not in effect (stale code ran)")
                    hard_fail = True
                if raw == 0:
                    print(f"  G3 ✗ {bench} has no 'As recorded' verbatim lines "
                          "— v2.4 superset rendering missing")
                    hard_fail = True
    # A model with no traces at all gated nothing. Printing PASS there reads as
    # "this backbone is fine" when it has simply never run (hy3 after its data
    # was archived) -- the same vacuous pass G1/G4 exist to prevent.
    if not gated_any:
        print(f"\n  ✗ no traces for '{model}' under {base} — nothing was gated; "
              "this is not a pass")
        hard_fail = True
    print("\n" + ("v2 GATE: FAIL — do not trust these numbers"
                  if hard_fail else "v2 GATE: PASS"))
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
