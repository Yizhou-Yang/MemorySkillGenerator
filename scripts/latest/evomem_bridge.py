"""Alias shim — the canonical implementation moved to `memlayer.bridge`
(the pip-installable SDK). Keeps harness imports and pickled CuratedMemory
stores (pickled under `scripts.latest.evomem_bridge`) resolving to the same
objects. Do not add code here."""
import sys as _sys
import memlayer.bridge as _impl
_sys.modules[__name__] = _impl
