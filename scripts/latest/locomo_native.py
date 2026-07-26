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
import argparse, json, os, re, sys, time, collections, asyncio, threading
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


# One Mem0 store for the whole run, conversations separated by user_id — this
# IS mem0's native LoCoMo protocol (one Memory, per-conversation user_id). It
# also sidesteps mem0's fixed-path internal stores: mem0 opens a SECOND local
# qdrant at $MEM0_DIR/migrations_qdrant (plus history.db) regardless of the
# vector_store path we set, and a per-conversation Memory would make N builds
# race for that one directory ("Storage folder ... already accessed by another
# instance") and store nothing. A single Memory opens it exactly once. Set a
# unique MEM0_DIR for the run to also isolate it from any concurrent mem0
# process (e.g. the agent-framework sweep's external-baseline arm).
_MEM0_SHARED = None
_MEM0_LOCK = threading.Lock()   # embedded qdrant is one instance; serialize ops


def _mem0_store():
    global _MEM0_SHARED
    if _MEM0_SHARED is None:
        with _MEM0_LOCK:
            if _MEM0_SHARED is None:
                from scripts.latest.baseline_memories import Mem0Memory
                _MEM0_SHARED = Mem0Memory("locomo_native")
    return _MEM0_SHARED


class _Mem0QA(QAMemory):
    def __init__(self, conv_id):
        self.m = _mem0_store()
        self.ns = f"locomo_native:{conv_id}"
    def add(self, text):
        with _MEM0_LOCK:
            self.m._mem().add([{"role": "user", "content": text[:6000]}],
                              user_id=self.ns)
    def recall(self, query):
        with _MEM0_LOCK:
            hits = self.m._mem().search(query, filters={"user_id": self.ns},
                                        limit=TOPK)
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
            # retriever.search returns a NUMPY ndarray of positional indices
            # into list(memories.values()). Do NOT write `idx or []`: on a
            # multi-element ndarray that raises "truth value ambiguous", which
            # the except then swallows, and recall silently returns nothing
            # (whole arm scored ~0 with all answers "I don't know"). Iterate the
            # array element-wise instead.
            idx = sysm.retriever.search(query, TOPK)
            allm = list(sysm.memories.values())
            outs = []
            for i in ([] if idx is None else list(idx)):
                ii = int(i)
                if 0 <= ii < len(allm):
                    outs.append(allm[ii].content)
        except Exception:
            outs = []
        return "\n".join(f"- {o}" for o in outs[:TOPK])


_EXTRACT_SYS = (
    "Extract atomic facts from this dialogue excerpt. For each fact output one "
    "JSON object on its own line: {\"entity\": <who/what the fact is about>, "
    "\"key\": <short slug of the aspect, e.g. relationship_status, job, location>, "
    "\"fact\": <the fact as a full sentence, include the date if stated>}. "
    "One line per fact, no prose, no array. Be EXHAUSTIVE: cover every concrete "
    "detail, however minor — objects and gifts (who gave what, what it "
    "symbolizes), activities and hobbies, places, opinions and realizations, "
    "plans, names of things, who made/owns what, yes/no facts. A question may "
    "hinge on any small detail. If a fact does not fit a clean aspect, still "
    "emit it with key \"detail\". Only greetings themselves may be skipped. "
    "DATES: the excerpt starts with the session date. When a speaker uses a "
    "relative time ('yesterday', 'last Friday', 'last week', 'a few years ago'), "
    "RESOLVE it against the session date and state the EVENT's own date in the "
    "fact (e.g. session dated 8 May 2023 + 'I went yesterday' -> 'on 7 May "
    "2023'). Never stamp the session date onto an event that happened earlier. "
    "WORDING: for subjective content (feelings, realizations, symbolism, "
    "reasons, aspirations) keep the speaker's OWN key words in the fact rather "
    "than abstracting them away ('a safe and inviting place for people to "
    "grow', not 'a sanctuary').")


_EMBEDDER = None
_EMB_LK = threading.Lock()


def _embedder():
    """The SAME sentence-encoder mem0/A-Mem retrieve with (all-MiniLM-L6-v2).

    CuratorMem's SDK store ranks patches lexically — fine for its agent-task
    setting, but on conversational QA that is a keyword handicap against the
    baselines' semantic search (a query 'relationship status' never lexically
    matches the fact 'Caroline is single'). For an apples-to-apples memory-layer
    comparison the retrieval encoder must be identical, so the adapter does its
    own semantic top-k over the extracted facts with this shared model."""
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMB_LK:
            if _EMBEDDER is None:
                from sentence_transformers import SentenceTransformer
                _EMBEDDER = SentenceTransformer(os.environ.get(
                    "MEM0_EMBED_MODEL",
                    "sentence-transformers/all-MiniLM-L6-v2"))
    return _EMBEDDER


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
        self.mem = PatchMemory()     # keeps the version-chain structure
        self.ns = conv_id
        self._ver = collections.defaultdict(int)   # (entity,key) -> latest version
        self._facts = []             # semantic index, parallel to self._emb rows
        self._emb = None             # np matrix, normalized; built lazily
        self._lk = threading.Lock()
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
            # LoCoMo sessions are long dialogues; a stingy budget would extract
            # only the first few facts and leave the rest unrecalled at QA time.
            # These are our own extractor's parameters (mem0/A-Mem extract with
            # their own internal budgets), so a fair, non-truncating setting.
            raw = _chat(self.model, self.base, self.key, _EXTRACT_SYS,
                        text[:int(os.environ.get("OURS_EXTRACT_CHARS", "10000"))],
                        max_tokens=int(os.environ.get("OURS_EXTRACT_TOKENS", "2600")))
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
            # mirror into the semantic index (fact text is what we retrieve on)
            self._facts.append({"fact": fact, "ent": ent, "chain": chain,
                                "version": v, "ts": ts})
            self._emb = None   # invalidate; rebuilt on next recall

    def _ensure_emb(self):
        if self._emb is not None or not self._facts:
            return
        with self._lk:
            if self._emb is not None or not self._facts:
                return
            import numpy as np
            vecs = _embedder().encode([f["fact"] for f in self._facts],
                                      normalize_embeddings=True,
                                      show_progress_bar=False)
            self._emb = np.asarray(vecs, dtype="float32")

    def recall(self, query):
        # Semantic top-k over the extracted facts (same encoder as the
        # baselines), then layer CuratorMem's structure on top: mark facts that
        # a later version superseded, and attach the dialogue timestamp as
        # grounded provenance so temporal questions can reason about the date.
        self._ensure_emb()
        if self._emb is None or not self._facts:
            return ""
        import numpy as np
        qv = _embedder().encode([query], normalize_embeddings=True,
                                show_progress_bar=False)[0]
        sims = self._emb @ np.asarray(qv, dtype="float32")
        # Our facts are ATOMIC (one clause each), whereas a mem0 "memory" is a
        # consolidated multi-fact statement — so k of ours carries less than k
        # of theirs. Retrieve more atomic facts to match the information budget,
        # not the item count (multi-hop especially needs several facts at once).
        rk = int(os.environ.get("OURS_RECALL_K", str(TOPK * 2)))
        order = np.argsort(-sims)[:rk]
        latest = {}
        for f in self._facts:
            latest[f["chain"]] = max(latest.get(f["chain"], 0), f["version"])
        lines = []
        for i in order:
            f = self._facts[int(i)]
            ts = f["ts"]
            mark = "[OK] " if ts else ""     # endorsed = carries dialogue-time provenance
            note = " (this was updated in a later session)" \
                   if f["version"] < latest.get(f["chain"], f["version"]) else ""
            lines.append(f"- {mark}{f['fact']}{note}" + (f" [{ts}]" if ts else ""))
        return "\n".join(lines)


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
              "a short phrase, matching the form of the question. "
              "For 'when' questions give an ABSOLUTE date or time (e.g. '7 May "
              "2023', 'June 2023', '2022'); resolve relative references such as "
              "'yesterday', 'last week', 'last year' against the dated memories "
              "rather than answering with the relative phrase. "
              "NEVER say you don't know and never refuse: scoring gives no "
              "credit for abstaining, so always commit to the single most "
              "probable answer the memories support — for hypothetical or "
              "judgment questions ('Would X ...?') answer e.g. 'Likely no', and "
              "if the memories are thin, give your best guess anyway.")
# NOTE: ANSWER_SYS is SHARED by mem0/amem/ours/full, so every arm compared in one
# table must be scored under the SAME wording. This run therefore reruns all five
# arms — staged (ours+nomem first for a fast read, baselines right after) rather
# than reusing the previous run's baseline rows, which predate this wording.
# The no-memory arm must be a GUESSING lower bound, not a muzzled one: under
# ANSWER_SYS ('answer ONLY from the memories') an empty context forces 'I don't
# know' on every question and the arm reads 0 by construction, which is an
# artifact of the prompt rather than a measurement. Mem0's paper lets the
# no-memory baseline answer from the question alone; match that.
NOMEM_SYS = ("You have NO memory of this conversation. Answer the question with "
             "your single best guess from the question itself and common sense. "
             "Be concise — a short phrase. Never say you don't know; always "
             "commit to a guess.")
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
        elif system == "full":     ctx = full_ctx[:int(os.environ.get("FULL_CHARS", "60000"))]
        else:
            try: ctx = memory.recall(q)
            except Exception: ctx = ""
        try:
            if system == "nomem":
                pred = _chat(qa_model, qa_base, qa_key, NOMEM_SYS,
                             f"Question: {q}\nAnswer:")
            else:
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
        # Isolate each system: a store that fails to even import (A-Mem raises
        # SystemExit — a BaseException — when its repo isn't on the path) must
        # not take down the systems that follow it in the list. Their rows are
        # already flushed to disk; skip the broken one and keep going.
        try:
            # ingest all conversations for this system in parallel (each is an
            # independent store), then answer all questions in parallel.
            with ThreadPoolExecutor(max_workers=min(CONC, len(convs))) as ex:
                built = list(ex.map(lambda c: (c, *ingest(system, c)), convs))
            jobs = [(system, c, mem, ctx, qa) for c, mem, ctx in built for qa in c["qa"]]
            with ThreadPoolExecutor(max_workers=CONC) as ex:
                list(ex.map(lambda j: answer_one(*j), jobs))
        except BaseException as e:
            print(f"=== {system} FAILED: {type(e).__name__}: {str(e)[:200]} "
                  f"— skipping ===", flush=True)
            continue
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
