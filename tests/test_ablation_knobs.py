"""Offline proof that each factorial-ablation knob changes what inject() serves.

Covers C_SIM_FLOOR, C_RANK and C_RENDER_RAW (appendix factorial arms). Runs
with no LLM and no network.

No LLM and no network: build a CuratedMemory, hand-populate its chain index
with entries, and drive the real inject(). Each knob is checked by diffing the
served block against the default configuration on identical state.
"""
import importlib, os, sys

KNOBS = ("C_SIM_FLOOR", "C_RANK", "C_RENDER_RAW")

def fresh(**env):
    for k in KNOBS:
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    for m in [m for m in list(sys.modules) if m.startswith("memlayer")]:
        del sys.modules[m]
    return importlib.import_module("memlayer.bridge")

class E:
    """Minimal stand-in for a stored experience (duck-typed, as inject reads it)."""
    def __init__(self, tid, ver, score, desc, endorsed, steps):
        self.task_id, self.version, self.score = tid, ver, score
        self.task_desc, self.timestamp = desc, float(ver)
        self.patch_history = [{"new_steps": ["s"] * steps, "removed_steps": []}]
        self.failure_taxonomy = {
            "verbatim_outcome": f"ANSWER-v{ver}",
            "causal_lesson": f"LESSON-v{ver} check the schema before the call",
            "avoidance_note": f"AVOID-v{ver}",
        }
        self.sys_stats = {"score_provenance": "env" if endorsed else "self_assessment",
                          "reuse_deltas": []}
        self.content = self.generalized_form = f"CONTENT-v{ver}"
        self.metadata, self.evidence = {}, desc
        # _concrete_approach reads this; without it _format_raw renders nothing
        # and every raw-path assertion below would pass vacuously.
        self.reasoning_trace = [f"TRACE-v{ver} ran the migration then checked counts"]
        self.action_commands, self.tool_sequence = [], []

DESC = "migrate the catalog table to the new sql dialect and verify row counts"
TASK = {"task_id": "t1", "description": DESC}

def build(mod, entries):
    m = mod.CuratedMemory.__new__(mod.CuratedMemory)     # skip LLM-touching __init__
    m.benchmark, m.top_k = "gaia", 3
    m._chain_of = {e.task_id: "t1" for e in entries}
    m._chain_entries = {"t1": list(entries)}
    m._served, m._served_keys, m._last_score, m._chain_base = {}, {}, {}, {}
    m._last_wc = None
    class _Lib:
        def retrieve_similar(self, *a, **k): return []
        def get_experience_weight(self, *a, **k): return 1.0
    class _SF: library = _Lib()
    m._sf = _SF()
    return m

ok = True
def check(name, cond, detail=""):
    global ok; ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))

# endorsed OLD version vs unendorsed NEW version: lineage and similarity disagree
# Insertion order IS the retriever order the sim foil preserves. Put the
# unendorsed newest first so similarity order (v3) and lineage order (endorsed
# v1) genuinely disagree — otherwise the foil is untestable.
ENTRIES = lambda: [E("t1", 3, 0.0, DESC, endorsed=False, steps=5),
                   E("t1", 1, 1.0, DESC, endorsed=True,  steps=0)]

b = fresh()
base = build(b, ENTRIES()).inject(TASK)
check("default inject is non-empty", bool(base.strip()), f"{len(base)} chars")

# --- C_RENDER_RAW -------------------------------------------------------------
r = fresh(C_RENDER_RAW="1")
raw_out = build(r, ENTRIES()).inject(TASK)
check("C_RENDER_RAW flag read", r._C_RENDER_RAW is True and b._C_RENDER_RAW is False)
check("C_RENDER_RAW still serves something (not vacuously empty)",
      bool(raw_out.strip()), f"{len(raw_out)} chars")
check("C_RENDER_RAW changes the served block", raw_out.strip() != base.strip())
check("curated block carries a lesson, raw foil does not",
      "LESSON" in base and "LESSON" not in raw_out)
check("raw foil still carries the underlying attempt", "TRACE" in raw_out)

# --- C_RANK -------------------------------------------------------------------
s = fresh(C_RANK="sim")
sim_out = build(s, ENTRIES()).inject(TASK)
check("C_RANK flag read", s._C_RANK == "sim" and b._C_RANK == "lineage")
first = lambda t: ("v1" if t.find("v1") >= 0 and (t.find("v3") < 0 or t.find("v1") < t.find("v3"))
                   else "v3")
check("lineage ranks the endorsed v1 first", first(base) == "v1")
check("C_RANK=sim serves a different order", first(sim_out) != first(base),
      f"lineage->{first(base)} sim->{first(sim_out)}")

# --- C_SIM_FLOOR ---------------------------------------------------------------
OFF = "completely unrelated wording about penguins and antarctic ice shelves"
off_entries = lambda: [E("t1", 1, 1.0, OFF, endorsed=True, steps=0)]
d = fresh()
ov = d._content_overlap(d._core_task(DESC), d._core_task(OFF))
lo = fresh(C_SIM_FLOOR="0.0")
hi = fresh(C_SIM_FLOOR="0.9")
check("C_SIM_FLOOR default 0.08 / reads env",
      abs(d._C_SIM_FLOOR - 0.08) < 1e-9 and lo._C_SIM_FLOOR == 0.0 and hi._C_SIM_FLOOR == 0.9)
lo_out = build(lo, off_entries()).inject(TASK)
hi_out = build(hi, off_entries()).inject(TASK)
check("off-topic entry has overlap below both defaults", ov < 0.08, f"overlap={ov:.3f}")
check("floor=0 serves the off-topic entry", bool(lo_out.strip()))
# Documented design: the floor is hygiene, so it may not empty a chain that has
# history. A single-entry chain is therefore served at ANY floor -- this asserts
# the documented behaviour rather than a filter the code deliberately does not have.
check("floor never empties a chain with history (hygiene, not filter)",
      bool(hi_out.strip()), "matches the paper's stated rule")
# Where the floor DOES bite: a multi-entry chain keeps only what clears it.
mixed = lambda: [E("t1", 1, 1.0, DESC, endorsed=True, steps=0),
                 E("t1", 2, 1.0, OFF,  endorsed=True, steps=0)]
d2, h2 = fresh(C_SIM_FLOOR="0.0"), fresh(C_SIM_FLOOR="0.5")
o_lo = build(d2, mixed()).inject(TASK)
o_hi = build(h2, mixed()).inject(TASK)
check("floor changes what a mixed chain serves", o_lo.strip() != o_hi.strip(),
      f"lo={len(o_lo)}ch hi={len(o_hi)}ch")

# --- render-kind telemetry ------------------------------------------------
b2 = fresh()
m2 = build(b2, ENTRIES())
m2.inject(TASK)
st = m2.pop_wc_stats() or {}
check("render_kind reported", st.get("render_kind") in
      ("curated", "repair_raw", "raw_fallback", "silence", "gated_raw", "empty"),
      str(st.get("render_kind")))
check("read-once semantics", m2.pop_wc_stats() is None)

print("\n" + ("ALL KNOBS VERIFIED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
