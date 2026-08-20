#!/usr/bin/env python3
"""Fill the paper's latency table from the benchmark's own JSON.

Typing four numbers per arm by hand is how a table drifts from its source, so
the table carries LAT_* placeholders and this writes the measured values in.
Re-running it after a new measurement updates the table and nothing else.

  python3 experiments_results/_paper/fill_latency.py [latency.json] [main.tex]
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    ROOT / "experiments_results" / "_paper" / "latency_hy3_tau2.json"
TEX = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper" / "main.tex"
KEY = {"curatormem": "C", "amem": "A", "mem0": "M"}


def main() -> int:
    data = json.loads(SRC.read_text())
    tex = TEX.read_text()
    rows = {r.get("arm"): r for r in data.get("rows", [])}
    missing = [a for a in KEY if a not in rows or rows[a].get("skipped")]
    if missing:
        print("refusing to fill: no usable measurement for %s" % ", ".join(missing),
              file=sys.stderr)
        return 1
    unserved = [a for a in KEY if not rows[a].get("served")]
    if unserved:
        print("refusing to fill: %s served 0 blocks, so its timings are a lookup "
              "that found nothing, not a read" % ", ".join(unserved), file=sys.stderr)
        return 1
    subs = {}
    for arm, k in KEY.items():
        r = rows[arm]
        for pct in ("50", "95", "99"):
            subs[f"LAT_{k}_{pct}"] = f"{r['p' + pct]:.2f}"
        subs[f"LAT_{k}_SERVED"] = "$%d/%d$" % (r["served"], r["n"])
    out, n = tex, 0
    for name, val in subs.items():
        out, c = re.subn(r"\b%s\b" % name, val, out)
        n += c
    if not n:
        print("no placeholders found -- table already filled?", file=sys.stderr)
        return 1
    TEX.write_text(out)
    print("filled %d placeholders from %s" % (n, SRC.name))
    for arm in KEY:
        r = rows[arm]
        print("  %-11s p50=%7.2f p95=%7.2f p99=%7.2f served=%d/%d"
              % (arm, r["p50"], r["p95"], r["p99"], r["served"], r["n"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
