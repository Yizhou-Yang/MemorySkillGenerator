#!/usr/bin/env python3
"""LoCoMo under the memory systems' OWN protocol (Mem0 / A-Mem paper setup),
not our iterative A/B/C framework.

Why this exists: LoCoMo is Mem0/A-Mem's flagship benchmark, and its task is
cross-session conversational QA — ingest a whole multi-session dialogue into
memory, then answer questions that require recalling facts from earlier
sessions. Forcing it into our "re-attempt the same task" loop makes memory of
prior *answers* irrelevant and scores the baselines as if they had no memory
(see the paper). So here every system runs the way its authors intended:

  ingest:  add each session's dialogue to the store, session by session
  answer:  for each question, retrieve top-k memories and let the LLM answer
           using ONLY the retrieved memories (no full transcript)
  score:   LLM-as-a-Judge (J, 0/1 correctness by an external judge) + token-F1

Systems compared behind one interface (make_qa_memory):
  mem0, amem  — their own add()/search()
  ours        — CuratorMem's store used purely as a memory layer
  full        — upper bound: whole transcript in context (no retrieval)
  nomem       — lower bound: answer with no memory

Run:
  BENCH_N=3 QA_MODEL=hy3 JUDGE_MODEL=grok-4.5 \
    python scripts/latest/locomo_native.py --systems mem0,amem,ours,nomem
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, collections, asyncio
from pathlib import Path

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

TOPK = int(os.environ.get("LOCOMO_TOPK", "10"))   # Mem0 uses s=10
OUT = Path(os.environ.get("LOCOMO_OUT",
           "experiments_results/locomo_native/hy3/results.jsonl"))


# ---- dataset -----------------------------------------------------------------
def load_conversations(n: int):
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    ds = load_dataset("KhangPTT373/locomo_preprocess", split="test")
    convs = []
    for i in range(min(n, len(ds))):
        r = ds[i]
        def parse(x):
            if isinstance(x, str):
                try: return json.loads(x)
                except Exception: return x
            return x
        sessions = parse(r.get("sessions")) or []
        qs = parse(r.get("questions")) or []
        ans = parse(r.get("answers")) or []
        cats = parse(r.get("category")) or []
        qa = []
        for j, q in enumerate(qs):
            qa.append({"question": q,
                       "answer": str(ans[j]) if j < len(ans) else "",
                       "category": str(cats[j]) if j < len(cats) else ""})
        convs.append({"conv_id": f"locomo_{i}", "sessions": sessions, "qa": qa})
    return convs


# ---- memory systems behind one QA interface ---------------------------------
class QAMemory:
    """add(text) to store a dialogue chunk; recall(query)->str for the block
    of top-k memories to hand the answerer. namespace = one conversation."""
    def add(self, text: str): ...
    def recall(self, query: str) -> str: ...


def make_qa_memory(system: str, conv_id: str) -> QAMemory:
    if system == "mem0":
        return _Mem0QA(conv_id)
    if system == "amem":
        return _AMemQA(conv_id)
    if system == "ours":
        return _OursQA(conv_id)
    raise ValueError(system)


class _Mem0QA(QAMemory):
    def __init__(self, conv_id):
        from scripts.latest.baseline_memories import Mem0Memory
        self.m = Mem0Memory("locomo_native")
        self.ns = f"locomo_native:{conv_id}"
    def add(self, text):
        m = self.m._mem_for(self.ns)
        m.add([{"role": "user", "content": text[:6000]}], user_id=self.ns)
    def recall(self, query):
        m = self.m._mem_for(self.ns)
        hits = m.search(query, filters={"user_id": self.ns}, limit=TOPK)
        items = hits.get("results", hits) if isinstance(hits, dict) else hits
        return "\n".join(f"- {h.get('memory') or h.get('text') or ''}"
                         for h in (items or []) if isinstance(h, dict))


class _AMemQA(QAMemory):
    def __init__(self, conv_id):
        from scripts.latest.baseline_memories import AMemMemory
        self.a = AMemMemory("locomo_native")
    def add(self, text):
        sysm = self.a._system()
        fn = getattr(sysm, "add_note", None) or getattr(sysm, "create_memory")
        fn(text[:6000])
    def recall(self, query):
        sysm = self.a._system()
        try:
            idx = sysm.retriever.search(query, TOPK)
            allm = list(sysm.memories.values())
            outs = [allm[int(i)].content for i in (idx or [])
                    if str(i).lstrip("-").isdigit() and 0 <= int(i) < len(allm)]
        except Exception:
            outs = []
        return "\n".join(f"- {o}" for o in outs[:TOPK])


_EXTRACT_SYS = (
    "Extract atomic facts from this dialogue excerpt. For each fact output one "
    "JSON object on its own line: {\"entity\": <who/what the fact is about>, "
    "\"key\": <short slug of the aspect, e.g. relationship_status, job, location>, "
    "\"fact\": <the fact as a full sentence, include the date if stated>}. "
    "One line per fact, no prose, no array. Skip greetings/small-talk.")


class _OursQA(QAMemory):
    """CuratorMem in its NATIVE structure, mapped onto conversation memory.

    A dialogue is not a bag of text: facts about one entity EVOLVE across
    sessions ("May: Caroline is single" -> "Aug: Caroline is dating"). That is
    exactly a patch version chain. So we (1) let the critic extract atomic
    facts (like mem0's extraction), (2) key each fact's chain on (entity,key)
    so a later fact about the same aspect SUPERSEDES the earlier one as a new
    version — lineage preserved, not destructively overwritten, (3) stamp the
    dialogue timestamp as grounded provenance, (4) retrieve chain-scoped and
    lineage-ordered (latest version first, older versions available for
    temporal questions). Mechanisms that need an execution loop (w_c reuse
    weight, avoidance-of-failed-attempts) have no analogue in static dialogue
    and are simply inactive here — stated as such."""
    def __init__(self, conv_id):
        from memlayer.vgr import PatchMemory
        self.mem = PatchMemory()
        self.ns = conv_id
        self._ver = collections.defaultdict(int)   # (entity,key) -> latest version
        # Extraction is CURATION — authored by the external critic (grok),
        # same author rule as the main method; the backbone only answers.
        self.model = os.environ.get("CRITIC_MODEL_ID",
                     os.environ.get("JUDGE_MODEL", "grok-4.5"))
        self.base = os.environ.get("CRITIC_BASE_URL",
                    os.environ.get("JUDGE_BASE_URL"))
        self.key = os.environ.get("CRITIC_API_KEY",
                   os.environ.get("JUDGE_API_KEY"))

    def add(self, text):
        from memlayer.vgr import Patch
        ts = ""
        m = re.match(r"\s*([0-9].*?[0-9]{4})", text)   # LoCoMo sessions start with a date line
        if m: ts = m.group(1)
        try:
            raw = _chat(self.model, self.base, self.key, _EXTRACT_SYS, text[:6000],
                        max_tokens=600)
        except Exception as e:
            print(f"[ours] extract skip: {str(e)[:60]}", flush=True); return
        for line in raw.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"): continue
            try: f = json.loads(line)
            except Exception: continue
            ent = str(f.get("entity", "")).strip().lower()
            key = str(f.get("key", "misc")).strip().lower()
            fact = str(f.get("fact", "")).strip()
            if not fact: continue
            chain = f"{self.ns}:{ent}:{key}"     # same aspect of same entity = one chain
            self._ver[chain] += 1
            v = self._ver[chain]
            prev = "" if v == 1 else "(supersedes earlier version)"
            self.mem.add(Patch(patch_id=f"{chain}#v{v}", chain_id=chain, version=v,
                               key=key, summary=f"{ent}: {fact}",
                               content_before=prev, content_after=fact,
                               rationale=ent, evidence=ts))   # ts = grounded provenance

    def recall(self, query):
        got = self.mem.retrieve(query, top_k=TOPK * 3)   # over-fetch, then dedupe by chain
        seen = {}
        for p in got or []:
            c = getattr(p, "chain_id", "")
            # lineage: keep the LATEST version per chain as the current answer,
            # but if an older version exists it means the fact changed — surface
            # both so temporal questions can reason about the evolution.
            seen.setdefault(c, []).append(p)
        lines = []
        for c, ps in list(seen.items())[:TOPK]:
            ps.sort(key=lambda p: getattr(p, "version", 1), reverse=True)
            cur = ps[0]
            ts = getattr(cur, "evidence", "")
            # Endorsement: only facts carrying an explicit dialogue timestamp
            # (objective provenance) wear the checkmark; undated ones render
            # neutrally — same two-key discipline as the main method.
            mark = "[OK] " if ts else ""
            lines.append(f"- {mark}{getattr(cur,'content_after','')}" + (f" [{ts}]" if ts else ""))
            for old in ps[1:2]:   # one prior version if the fact evolved
                lines.append(f"    (earlier: {getattr(old,'content_after','')}"
                             + (f" [{getattr(old,'evidence','')}]" if getattr(old,'evidence','') else "") + ")")
        return "\n".join(lines[:TOPK * 2])


# ---- LLM calls ---------------------------------------------------------------
def _chat(model, base, key, system, user, max_tokens=400):
    from openai import OpenAI
    cli = OpenAI(base_url=base, api_key=key or "x", timeout=180)
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    req = {"model": model, "messages": msgs, "max_tokens": max_tokens}
    if os.environ.get("QA_REASONING_EFFORT"):
        req["reasoning_effort"] = os.environ["QA_REASONING_EFFORT"]
    r = cli.chat.completions.create(**req)
    m = r.choices[0].message
    return (m.content or "").strip() or (getattr(m, "reasoning_content", "") or "").strip()


ANSWER_SYS = ("Answer the question using ONLY the memories below. Be concise — "
              "a short phrase or sentence. If the memories don't contain the "
              "answer, say you don't know.")
JUDGE_SYS = ("You are a strict grader. Given a question, the gold answer, and a "
             "candidate answer, output 1 if the candidate is correct (same "
             "meaning as gold), else 0. Output ONLY 0 or 1.")


def f1(pred, gold):
    def toks(s): return re.findall(r"[a-z0-9]+", s.lower())
    p, g = toks(pred), toks(gold)
    if not p or not g: return 0.0
    common = collections.Counter(p) & collections.Counter(g)
    ov = sum(common.values())
    if ov == 0: return 0.0
    prec, rec = ov/len(p), ov/len(g)
    return 2*prec*rec/(prec+rec)


def judge(model, base, key, q, gold, pred):
    try:
        out = _chat(model, base, key, JUDGE_SYS,
                    f"Question: {q}\nGold: {gold}\nCandidate: {pred}\nScore:",
                    max_tokens=4)
        m = re.search(r"[01]", out)
        return int(m.group()) if m else 0
    except Exception:
        return 0


# ---- main --------------------------------------------------------------------
def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="mem0,amem,ours,nomem")
    args = ap.parse_args()
    n = int(os.environ.get("BENCH_N", "3"))
    qa_model = os.environ.get("QA_MODEL", "hy3")
    qa_base = os.environ.get("OPENAI_API_BASE")
    qa_key = os.environ.get("OPENAI_API_KEY")
    j_model = os.environ.get("JUDGE_MODEL", "grok-4.5")
    j_base = os.environ.get("JUDGE_BASE_URL", os.environ.get("CRITIC_BASE_URL"))
    j_key = os.environ.get("JUDGE_API_KEY", os.environ.get("CRITIC_API_KEY"))

    import threading
    from concurrent.futures import ThreadPoolExecutor
    CONC = int(os.environ.get("LOCOMO_CONC", "16"))
    convs = load_conversations(n)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fout = open(OUT, "a"); wlock = threading.Lock()
    agg = collections.defaultdict(lambda: {"J": 0, "F1": 0.0, "n": 0})
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    def ingest(system, conv):
        memory = None if system in ("nomem", "full") else make_qa_memory(system, conv["conv_id"])
        transcript = []
        # sessions of ONE conversation stay ordered (later memory builds on
        # earlier); different (system, conv) units run in parallel below.
        for sess in conv["sessions"]:
            txt = sess if isinstance(sess, str) else json.dumps(sess)
            transcript.append(txt)
            if memory is not None:
                try: memory.add(txt)
                except Exception as e: print(f"[{system}] add skip: {str(e)[:80]}", flush=True)
        return memory, "\n\n".join(transcript)

    def answer_one(system, conv, memory, full_ctx, qa):
        q, gold = qa["question"], qa["answer"]
        if system == "nomem":      ctx = "(no memory available)"
        elif system == "full":     ctx = full_ctx[:24000]
        else:
            try: ctx = memory.recall(q)
            except Exception: ctx = ""
        try:
            pred = _chat(qa_model, qa_base, qa_key, ANSWER_SYS,
                         f"Memories:\n{ctx or '(none)'}\n\nQuestion: {q}\nAnswer:")
        except Exception: pred = ""
        J = judge(j_model, j_base, j_key, q, gold, pred); F = f1(pred, gold)
        with wlock:
            a = agg[system]; a["J"] += J; a["F1"] += F; a["n"] += 1
            fout.write(json.dumps({"system": system, "conv": conv["conv_id"],
                "category": qa["category"], "q": q, "gold": gold,
                "pred": pred[:300], "J": J, "F1": round(F, 3)}, ensure_ascii=False) + "\n")
            fout.flush()

    for system in systems:
        # ingest all conversations for this system in parallel (each is an
        # independent store), then answer all questions in parallel.
        with ThreadPoolExecutor(max_workers=min(CONC, len(convs))) as ex:
            built = list(ex.map(lambda c: (c, *ingest(system, c)), convs))
        jobs = [(system, c, mem, ctx, qa) for c, mem, ctx in built for qa in c["qa"]]
        with ThreadPoolExecutor(max_workers=CONC) as ex:
            list(ex.map(lambda j: answer_one(*j), jobs))
        a = agg[system]
        print(f"=== {system}: J={a['J']/max(1,a['n'])*100:.1f} "
              f"F1={a['F1']/max(1,a['n'])*100:.1f} (n={a['n']}) ===", flush=True)
    fout.close()
    print("\n=== SUMMARY (LoCoMo native protocol) ===")
    for system, a in agg.items():
        print(f"  {system:6s}: J={a['J']/max(1,a['n'])*100:.1f} "
              f"F1={a['F1']/max(1,a['n'])*100:.1f} (n={a['n']})")


if __name__ == "__main__":
    run()
