"""memlayer SDK — packaging-boundary tests.

Covers the three guarantees the SDK extraction must not break:
  1. alias identity — old harness paths (src.latest.*, scripts.latest.
     evomem_bridge) resolve to the SAME module objects as memlayer.*, so
     harness imports and previously pickled stores keep working;
  2. the facade round-trip (record → inject → save → load) works with
     injected stub LLMs and no network;
  3. the standalone fallback (memlayer.llm) fails loud, not silent, when no
     endpoint is configured.
"""
from __future__ import annotations

import asyncio
import os

import pytest

# The reviewer pen defaults to the external critic (fail-loud without HY3);
# these tests inject stub LLMs directly, so pin the authorship knob to the
# self-contained setting BEFORE memlayer imports read it.
os.environ.setdefault("METADATA_AUTHOR", "backbone")


def _stub_llm(prompt: str) -> str:
    return (
        '{"quality_score": 7, "total": 7, "verdict": "inject", '
        '"causal_lesson": "stub lesson long enough to clear the weak-lesson floor", '
        '"generalized_steps": "step one then step two", '
        '"avoidance_note": "", "transferability": "high", "refined": true}'
    )


def test_alias_identity():
    import memlayer.bridge as mb
    import memlayer.experience as me

    import scripts.latest.evomem_bridge as eb
    import src.latest.experience as se

    assert se is me, "src.latest.experience must alias memlayer.experience"
    assert eb is mb, "scripts.latest.evomem_bridge must alias memlayer.bridge"
    # pickle-path compatibility: same class objects through both names
    assert eb.CuratedMemory is mb.CuratedMemory
    assert se.Experience is me.Experience

    from src.latest import SkillForgeLatest as S1
    from memlayer.forge import SkillForgeLatest as S2
    assert S1 is S2


def test_experience_catalog_fields():
    from memlayer.experience import Experience

    e = Experience("t1", "d", [], [], "success", 1.0, [], [], "")
    assert e.sys_stats == {} and e.meta_author == ""


def test_record_inject_roundtrip(tmp_path):
    from memlayer import MemoryLayer

    mem = MemoryLayer(llm=_stub_llm, critic=_stub_llm, domain="t")
    for i, (content, score) in enumerate(
            [("tried X, partially worked", 0.6), ("tried X properly, worked", 1.0)]):
        asyncio.run(mem.arecord(content, chain_id="c1", task="do the thing",
                                task_id="t1", score=score))
    assert len(mem) == 2

    block = mem.inject("do the thing", chain_id="c1", task_id="t1")
    assert block, "chain has history — inject must render a non-empty block"

    man = mem.manifest()
    assert man["chains"] == 1 and man["entries"] == 2

    p = tmp_path / "store.pkl"
    mem.save(p)
    mem2 = MemoryLayer.load(p, llm=_stub_llm)
    assert len(mem2) == 2
    assert mem2.inject("do the thing", chain_id="c1", task_id="t1")


def test_standalone_llm_fails_loud(monkeypatch):
    from memlayer import llm as ml

    for k in ("HY3_BASE_URL", "HY3_API_KEY", "CRITIC_BASE_URL", "CRITIC_API_KEY",
              "OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENAI_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError):
        ml.llm_critic_fn("ping")
    with pytest.raises(RuntimeError):
        ml.llm_review_fn("ping")
