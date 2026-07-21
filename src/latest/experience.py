"""Alias shim — the canonical implementation moved to `memlayer.experience`
(the pip-installable SDK). This file keeps old imports AND previously pickled
stores (classes pickled under `src.latest.experience`) resolving to the same
objects. Do not add code here."""
import sys as _sys
import memlayer.experience as _impl
_sys.modules[__name__] = _impl
