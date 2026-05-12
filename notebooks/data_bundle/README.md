# `data_bundle/` — vendored replay data

Self-contained data needed by the replay-only notebook paths
(`01_*.ipynb`, `02a_*.ipynb`, `03a_*.ipynb`, and the `BACKEND="replay"`
branch of any realtime notebook). Total footprint ~220 KB.

`_demo_utils.py` reads this folder first; if absent, it falls back to
the source paths under `../../RF-branching/` for developers working
inside the full repo.

## Contents

| File | Source | Purpose |
|---|---|---|
| `psp_01.json` | `RF-branching/slides/qcg_vs_cg_demo/data/psp_01.json` | First per-PSP bundle (graph, duals, Dirac columns, LP columns, layout) for nb 02a |
| `psp_02.json` | `RF-branching/slides/qcg_vs_cg_demo/data/psp_02.json` | Second per-PSP bundle (later CG iteration with differentiated duals) |
| `er_n20_p70_s0/qcg/raw_samples/call_*.pkl` | `RF-branching/instances/er_n20_p70_s0/qcg/raw_samples/` | Six recorded Dirac-3 cloud responses from a full quantum CG run on ER(20, 0.7, seed=0); replayed by `ReplayDiracOracle` |
| `er_n20_p70_s0/qcg/raw_samples/index.jsonl` | same | Index of the calls above (one row per call) |
| `er_n20_p70_s0/cg/raw_samples/*` | `RF-branching/instances/er_n20_p70_s0/cg/raw_samples/` | Companion classical-LP recorded run on the same instance |

## Instance identity

The bundled instance is **ER(20, 0.7, seed=0)** — 20 vertices, 131 edges,
chromatic number χ = 8. This is the smallest instance from the QCG-vs-CG
benchmarking study and is small enough that a full Dirac-3 run replays
in ~200 ms (vs ~9 minutes on the original device).

## Adding your own runs

Realtime notebooks (`02b`, `03b`) with `SAVE_RUN=True` write to
`notebooks/runs/<UTC-timestamp>_<label>/raw_samples/` in the **same
schema** as the directories here. Once captured, you can replay your run
offline:

```python
import _demo_utils as U
oracle = U.replay_oracle_from_run("notebooks/runs/<your-run>")
```
