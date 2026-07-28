# memlayer — curated, manifested, append-only agent memory

A memory layer for LLM agents with a small imperative API (record / inject /
search) and three guarantees most memory layers do not make:

1. **Append-only.** `record()` and additive curation only. No delete, no
   merge, no rewrite — an entry that looks useless today is still retrievable
   after the rollback that makes it correct again.
2. **A manifest over the store.** Chains are partitions, version history is
   lineage, effectiveness weights are per-entry statistics. Reads are
   partition-pruned: **O(chain), flat as the store grows** (measured flat from
   10³ to 10⁵ patches; see `scripts/latest/manifest_bench.py`). Lineage makes
   `time_travel()` a query, not a reconstruction effort.
3. **Dose-budgeted, never-silent rendering.** The injected block is capped
   (`C_INJECT_BUDGET_CH`, default 900 chars), replays **every** prior attempt
   verbatim before spending leftover budget on annotations, and degrades to
   raw entries rather than silence when curated channels come up empty.

LLM is **optional**: without one, the layer is a pure append-only patch
memory; with one, entries are additionally refined, critiqued, and enriched
at write time (additively — curation never edits the underlying record).

## Quickstart

```python
from memlayer import MemoryLayer

mem = MemoryLayer(llm=my_llm_fn)             # or llm=None for append-only
mem.record("v1 API rejected; v2 needs a partition spec",
           chain_id="deploy-42",
           task="register table with catalog", score=0.4)

block = mem.inject("how do I register the table?", chain_id="deploy-42")
prompt = block + "\n\n" + user_task          # prepend as supporting context

mem.manifest()                                # {chains, entries, per_chain}
mem.time_travel("deploy-42", as_of=1)         # context as of entry 1
mem.save("store.pkl"); MemoryLayer.load("store.pkl")
```

## API

| Call | What it does |
|---|---|
| `record(content, *, chain_id, task, score, metadata)` | append one attempt (never overwrites); `arecord` for async |
| `inject(query, *, chain_id)` | rendered, dose-budgeted context block ("" if no history — never invented) |
| `search(query, *, chain_id, k)` | raw entry metadata, newest first |
| `manifest(chain_id=None)` | partitions / lineage / statistics |
| `time_travel(chain_id, as_of=k)` | the block as it existed after k entries |
| `save(path)` / `load(path, llm=)` | pickle persistence (LLM handle re-injected on load) |

Knobs (env): `C_INJECT_BUDGET_CH` (dose), `C_CRITIC_GATE` (quality gate),
`C_RAW_FALLBACK` (never-silent), `C_PAGE_KEEP` (weak-compaction paging,
0 = off = append-only).

### Retrieval

`PatchMemory` ranks lexically by default (zero dependencies). For corpora where
paraphrase matters — conversational memory, say — pass an encoder and it ranks
semantically instead; chain scoping and the recency tie-break are unchanged.

```python
from memlayer.vgr import PatchMemory
enc = SentenceTransformer("all-MiniLM-L6-v2")
mem = PatchMemory(embedder=lambda ts: enc.encode(ts, normalize_embeddings=True))
```

Minimal deps (no LLM mode): `rapidfuzz json_repair python-dotenv numpy
requests`; `sentence-transformers` optional (falls back to TF-cosine).

## Scope notes

- This is a thin facade over the experiment code in
  `scripts/latest/evomem_bridge.py` — the exact read/write paths the paper's
  arms run, no forked logic.
- `chain_id` scopes memory to iterations of the same task/session by design;
  cross-chain transfer is deliberately out of scope here.
- No benchmark numbers in this README by policy; see the paper.
