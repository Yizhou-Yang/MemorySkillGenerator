# Traces, laid out the way the paper reads

One directory per Table 1 row, one file per arm:

    tab1/<benchmark>/<backbone>/{raw,patched,amem,mem0,curatormem}.jsonl

Run-shaped storage hid which arms share a draw, and that is the thing a reader
needs. GAIA/DeepSeek takes four arms from the `dscb` sweep and its curated arm
from `dscb4_gaia`; tau2/HY3 takes Raw and CuratorMem from a paired rerun and its
other three from an earlier sweep. Under the old paths nothing said so, and a
row's ordering means something quite different when its arms come from one draw
than when they come from three.

`PROVENANCE.tsv` carries that column: benchmark, backbone, arm, rows, source run,
snapshot. `_paper/verify.py` recomputes every printed cell from these files.

Two rows keep their own shape. LoCoMo under HY3 and GPT-5.5 is scored
per-question (judge correctness and token-F1 over 497 questions) rather than
per-task, so it stays under `locomo_native/` in its native format; forcing it
into the arm-file schema would drop the per-question structure the metric needs.
They are listed in PROVENANCE.tsv all the same.
