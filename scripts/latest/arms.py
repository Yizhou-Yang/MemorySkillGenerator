#!/usr/bin/env python3
"""Canonical arm keys — ONE place for group naming.

Meaningful, disclosure-safe keys replace the old A/B/C internals (the legacy
keys leaked project codenames into public traces):

    no_mem         (was A_baseline)   agent with no memory
    raw_patch      (was B_evomem)     naive append-only patch replay, the
                                      \\patchmem baseline
    curated_patch  (was C_gpr)        the full method
    no_mem_passk_s<i>                 i-th independent sample of the pass@k
                                      variance/compute control (no memory)
    mem0 / amem / ...                 external baseline memory systems

Writers emit canonical keys; readers call norm_group() so traces written
before the rename (legacy keys) keep pairing with new rows.
"""

CANON = {"A": "no_mem", "B": "raw_patch", "C": "curated_patch"}

_LEGACY = {
    "A_baseline": "no_mem",
    "B_evomem": "raw_patch",
    "C_gpr": "curated_patch",
}


def norm_group(g: str) -> str:
    """Map any historical group key to its canonical name (idempotent)."""
    return _LEGACY.get(g, g)
