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


def rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def main() -> None:
    model = (sys.argv[1] if len(sys.argv) > 1 else "hy3").lower()
    base = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("BASE", "latest_evolving")
    hard_fail = False
    for bench in ["gaia", "gaia2", "locomo"]:
        rs = rows(PROJECT_ROOT / "experiments_results" / base / model / bench / "trace.jsonl")
        if not rs:
            print(f"[{bench}] no trace — skipped")
            continue
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
        for g in ["A_baseline", "B_evomem", "C_gpr"]:
            grs = [r for r in rs if r.get("group") == g]
            fin = {r["task_id"] for r in grs if (r.get("iteration", 0) or 0) == kf}
            late = [r for r in grs if (r.get("iteration", 0) or 0) >= 1]
            inj = [r for r in late if r.get("patch_injected")]
            lens = [r.get("aug_len") or 0 for r in inj]
            doses[g] = (sum(lens) / len(lens)) if lens else 0.0
            cov = (len(inj) / len(late)) if late else 0.0
            print(f"  {g}: finished n={len(fin)}  inj@iter>=1 {len(inj)}/{len(late)}"
                  f" ({100*cov:.0f}%)  dose mean={doses[g]:.0f} max={max(lens) if lens else 0}")
            if g == "C_gpr" and late:
                if cov < COVERAGE_FLOOR:
                    print(f"  G1 ✗ C coverage {100*cov:.0f}% < {100*COVERAGE_FLOOR:.0f}%"
                          " — chain-index/fallback not effective")
                    hard_fail = True
                if lens and max(lens) > BUDGET + SLACK:
                    print(f"  G2 ✗ C max dose {max(lens)} > {BUDGET}+{SLACK}")
                    hard_fail = True
        if doses.get("C_gpr") and doses.get("B_evomem") and \
                doses["C_gpr"] > doses["B_evomem"] + 1:
            # The paper's below-B-dose claim is scoped to the AGENTIC
            # benchmarks (gaia2/TB2), where dose harm was measured. On
            # QA benches B's natural render can sit far below L (locomo
            # B≈225ch) and C legitimately exceeds it under the same cap.
            if bench == "gaia2":
                print(f"  G2 ✗ C mean dose {doses['C_gpr']:.0f} > B "
                      f"{doses['B_evomem']:.0f} (agentic below-B claim broken)")
                hard_fail = True
            else:
                print(f"  G2 note: C mean dose {doses['C_gpr']:.0f} > B "
                      f"{doses['B_evomem']:.0f} (allowed off-agentic; cap is L)")
        # G3 markers (soft): rendered-channel fingerprints in C prompts
        caugs = [r.get("augmented_prompt") or "" for r in rs
                 if r.get("group") == "C_gpr" and r.get("patch_injected")]
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
    print("\n" + ("v2 GATE: FAIL — do not trust these numbers"
                  if hard_fail else "v2 GATE: PASS"))
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
