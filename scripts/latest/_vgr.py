"""Import shim for the VGR engine.

Importing ``src.latest.vgr`` normally runs ``src/latest/__init__.py``, which
eagerly pulls optional deps (rapidfuzz, sentence-transformers, ...). The engine
itself is pure stdlib. This shim returns the engine whether or not those deps
are installed, so the tests and the simulation backend run anywhere. In a normal
(deps-installed) environment it resolves to the real package module.
"""
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from src.latest import vgr as _engine  # type: ignore
except Exception:
    if "latest_vgr_engine" in sys.modules:
        _engine = sys.modules["latest_vgr_engine"]
    else:
        _spec = importlib.util.spec_from_file_location(
            "latest_vgr_engine", _ROOT / "src" / "latest" / "vgr.py")
        _engine = importlib.util.module_from_spec(_spec)
        sys.modules["latest_vgr_engine"] = _engine
        _spec.loader.exec_module(_engine)

Patch = _engine.Patch
VerificationResult = _engine.VerificationResult
GroundedPatch = _engine.GroundedPatch
Verifier = _engine.Verifier
PatchMemory = _engine.PatchMemory
ground = _engine.ground
render_patches_plain = _engine.render_patches_plain
render_patches_grounded = _engine.render_patches_grounded
_lexical_sim = _engine._lexical_sim
repair_hint = _engine.repair_hint
ActionOutcome = _engine.ActionOutcome
ResolveConfig = _engine.ResolveConfig
ResolveTrace = _engine.ResolveTrace
resolve_and_solve = _engine.resolve_and_solve
_patch_line = _engine._patch_line
