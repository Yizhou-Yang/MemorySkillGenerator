"""Offline tests for Grounded Patch Resolution (Method C).

These run with stdlib only (no LLM/Docker), and prove the properties the paper
relies on:
  1. Additive invariant: C's injected context is a strict superset of B's.
  2. Grounding never drops a patch (it only annotates).
  3. The env-probe verifier reads ground truth correctly.
  4. repair_hint fires only toward an environment-confirmed state.
  5. The chain simulation respects C >= B per task and shows a larger C-B gap
     at chain level than at task level.
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.latest._vgr import (  # noqa: E402
    Patch, PatchMemory, GroundedPatch, ground, render_patches_plain,
    render_patches_grounded, repair_hint, ActionOutcome, _patch_line,
)
from scripts.latest.terminal_verifier import EnvProbeVerifier  # noqa: E402
from scripts.latest.vgr_experiment import _simulate, SimParams  # noqa: E402


def _patch(pid, key, after, version=1, probe="check", signal="OK", neg=False):
    return Patch(patch_id=pid, chain_id="c1", version=version, key=key,
                 summary=f"{key} changed", content_before="old",
                 content_after=after, rationale="env update",
                 probe=probe, expected_signal=signal, is_negative=neg)


def _fake_exec(mapping):
    async def _exec(cmd):
        return mapping.get(cmd, ("", 1))
    return _exec


def test_additive_superset():
    """C's grounded injection must contain B's plain injection line-for-line."""
    patches = [_patch("p1", "output_path", "/out/v2"),
               _patch("p2", "branch", "main", version=2)]
    plain = render_patches_plain(patches)
    grounded = render_patches_grounded([GroundedPatch(p) for p in patches])
    # every B patch line appears verbatim in C's render
    for p in patches:
        assert _patch_line(p) in plain
        assert _patch_line(p) in grounded
    print("test_additive_superset: PASS")


def test_grounding_never_drops():
    patches = [_patch("p1", "k1", "a", probe="c1", signal="OK"),
               _patch("p2", "k2", "b", probe="c2", signal="NOPE")]
    verifier = EnvProbeVerifier(_fake_exec({"c1": ("OK here", 0),
                                            "c2": ("something else", 0)}))
    grounded = asyncio.run(ground(patches, verifier, {}))
    assert len(grounded) == len(patches)          # nothing filtered
    assert grounded[0].verification.applies is True    # signal present
    assert grounded[1].verification.applies is False   # signal absent
    print("test_grounding_never_drops: PASS")


def test_verifier_signal_and_rc():
    # expected_signal present -> applies True regardless of rc
    v = EnvProbeVerifier(_fake_exec({"probe": ("...OK...", 3)}))
    r = asyncio.run(v.verify(_patch("p", "k", "a", probe="probe", signal="OK"), {}))
    assert r.applies is True and r.confidence == 1.0
    # no expected_signal -> falls back to returncode==0
    v_ok = EnvProbeVerifier(_fake_exec({"pz": ("done", 0)}))
    r2 = asyncio.run(v_ok.verify(Patch("p", "c", 1, "k", "s", "o", "n",
                                       probe="pz", expected_signal=""), {}))
    assert r2.applies is True           # rc 0 -> applies
    v_bad = EnvProbeVerifier(_fake_exec({"pz": ("boom", 1)}))
    r3 = asyncio.run(v_bad.verify(Patch("p", "c", 1, "k", "s", "o", "n",
                                        probe="pz", expected_signal=""), {}))
    assert r3.applies is False          # rc 1 -> does not apply
    # absent probe -> UNKNOWN, never dropped
    r4 = asyncio.run(v_ok.verify(Patch("p", "c", 1, "k", "s", "o", "n",
                                       probe="", expected_signal=""), {}))
    assert r4.applies is None
    print("test_verifier_signal_and_rc: PASS")


def test_repair_fires_only_when_grounded():
    confirmed = GroundedPatch(_patch("p1", "branch", "main"))
    grounded_true = asyncio.run(ground(
        [confirmed.patch], EnvProbeVerifier(_fake_exec({"check": ("OK", 0)})), {}))
    # failed action + a confirmed current state -> hint
    hint = repair_hint(ActionOutcome(passed=False, observation="tests failed"),
                       grounded_true)
    assert hint is not None and "branch" in hint
    # passed action -> never repairs
    assert repair_hint(ActionOutcome(passed=True), grounded_true) is None
    # failed action but nothing grounded-true -> no blind repair
    grounded_false = asyncio.run(ground(
        [_patch("p2", "k", "x", probe="check", signal="OK")],
        EnvProbeVerifier(_fake_exec({"check": ("mismatch", 1)})), {}))
    assert repair_hint(ActionOutcome(passed=False), grounded_false) is None
    print("test_repair_fires_only_when_grounded: PASS")


def test_sim_monotonic_and_chain_amplifies():
    res = _simulate(SimParams(n_chains=400, n_versions=4, seed=7))
    ta, ca = res["task_accuracy"], res["chain_accuracy"]
    # additive invariant must hold exactly
    assert res["diagnostics"]["C_ge_B_violations"] == 0
    # ordering A < B < C
    assert ta["A"] < ta["B"] < ta["C"], ta
    assert ca["A"] < ca["B"] < ca["C"], ca
    # chain-level C-B gap exceeds task-level C-B gap (compounding advantage)
    assert (ca["C"] - ca["B"]) > (ta["C"] - ta["B"]), (ca, ta)
    # B genuinely regresses on some tasks (the failure mode C removes)
    assert res["diagnostics"]["B_regressions"] > 0
    print("test_sim_monotonic_and_chain_amplifies: PASS")


if __name__ == "__main__":
    test_additive_superset()
    test_grounding_never_drops()
    test_verifier_signal_and_rc()
    test_repair_fires_only_when_grounded()
    test_sim_monotonic_and_chain_amplifies()
    print("\nALL VGR TESTS PASSED")
