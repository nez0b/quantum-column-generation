# `data_bundle/` — vendored replay data

Self-contained data needed by the replay-only notebook paths
(`01_*.ipynb`, `02a_*.ipynb`, `03a_*.ipynb`, and the `BACKEND="replay"`
branch of any realtime notebook). Total footprint ~1.3 MB.

`_demo_utils.py` reads this folder first; if absent, it falls back to
the source paths under `../../RF-branching/` for developers working
inside the full repo.

## Contents

| File | Source | Purpose |
|---|---|---|
| `psp_01.json` / `psp_02.json` | `RF-branching/slides/qcg_vs_cg_demo/data/` | Two per-PSP bundles (graph, duals, Dirac+LP columns, layout) for the 02a walkthrough; n=20 only |
| `er_n20_p70_s0/qcg/raw_samples/call_*.pkl` (6 calls) | `RF-branching/instances/er_n20_p70_s0/qcg/raw_samples/` | Recorded Dirac-3 responses for the n=20 quantum CG run |
| `er_n20_p70_s0/qcg/raw_samples/index.jsonl` | same | Index of the calls above |
| `er_n20_p70_s0/cg/raw_samples/*` (4 calls) | `RF-branching/instances/er_n20_p70_s0/cg/raw_samples/` | Companion classical-LP run on the same instance |
| `er_n50_p30_s0/qcg/raw_samples/call_*.pkl` (14 calls) | `RF-branching/instances/er_n50_p30_s0/qcg/raw_samples/` | Recorded Dirac-3 responses for the n=50 quantum CG run (the "quantum beats classical" example) |
| `er_n50_p30_s0/qcg/raw_samples/index.jsonl` | same | Index for the n=50 calls |

## Bundled instances

| Instance | n | m | Classical CG χ (live) | Quantum CG χ (replay) | Optimal |
|---|---:|---:|:-:|:-:|:-:|
| `er_n20_p70_s0` | 20 | 131 | 8 | 8 | 8 |
| `er_n50_p30_s0` | 50 | 362 | 12 | **9** | 6 (MILP/Hexaly) |

On `er_n50_p30_s0` the bundled Dirac replay beats classical column
generation by **3 colors** — a faithful reproduction of the headline
result from the benchmark study. The optimum is 6 colors (MILP) but the
LP relaxation lower bound from CG can't close that gap on this
instance; the χ=9 vs χ=12 separation is the value-add of the quantum
sampler.

Each saved CallRecord contains the full `node_list` and `graph_edges`,
so `_demo_utils.load_bundled_instance(id)` reconstructs the graph
without needing a separate PSP JSON.

## Adding your own runs

Realtime notebooks (`02b`, `03b`) with `SAVE_RUN=True` write to
`notebooks/runs/<UTC-timestamp>_<label>/raw_samples/` in the **same
schema** as the directories here. Once captured, you can replay your run
offline:

```python
import _demo_utils as U
oracle = U.replay_oracle_from_run("notebooks/runs/<your-run>")
```
