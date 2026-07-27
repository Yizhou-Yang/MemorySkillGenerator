"""Alias shim — the canonical implementation moved to `memlayer.vgr`
(the pip-installable SDK). This file keeps old imports AND previously pickled
stores (classes pickled under `src.latest.vgr`) resolving to the same
objects. Do not add code here."""
import sys as _sys
import memlayer.vgr as _impl
_sys.modules[__name__] = _impl
