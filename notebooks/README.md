# QCG Tutorial Bundle

End-to-end tutorial **and** companion slide decks for QCi's quantum
column-generation pipeline for graph coloring. Self-contained: notebooks
run on bundled replay data without device access; slides ship as
pre-built PDFs.

```
notebooks/
├── 01_motzkin_straus.ipynb              ← theory: MS QP + Dirac as sampler
├── 02a_column_generation_replay.ipynb   ← CG walkthrough on bundled data
├── 02b_column_generation_realtime.ipynb ← CG on your graph + live Dirac
├── 03a_application_demo_replay.ipynb    ← six-solver pipeline on bundled data
├── 03b_application_demo_realtime.ipynb  ← six-solver pipeline on your graph
├── 04_tutorial_end_to_end.ipynb         ← single-file end-to-end tutorial
├── _demo_utils.py                       ← shared helpers
├── pyproject.toml / .python-version     ← portable env (uv-managed)
├── data_bundle/                         ← vendored replay data (~220 KB)
├── slides/                              ← three beamer-qci PDFs + sources
└── runs/                                ← gitignored; live-mode captures
```

---

## Quick start

```bash
cd notebooks
uv venv                # creates notebooks/.venv (Python 3.12)
uv sync                # cloud-only deps
uv sync --extra direct # add eqc-direct for on-prem mode (optional)
uv run jupyter lab .
```

Or from the parent repo root, sharing the project venv:

```bash
uv run jupyter lab notebooks/
```

Both work because `_demo_utils.py` resolves paths from `__file__` and
adds `../src` to `sys.path` if the editable install isn't present.

---

## Notebook index

| Notebook | Topic | Live API needed? |
|---|---|---|
| `01_motzkin_straus.ipynb` | Motzkin-Straus QP and Dirac-3 as a clique sampler. SLSQP vs Dirac on a small graph. | Optional (replay default) |
| `02a_column_generation_replay.ipynb` | Column generation step-by-step on bundled ER(20, 0.7) data. | No |
| `02b_column_generation_realtime.ipynb` | Same content on **your** graph + live Dirac. Saves run for offline replay. | Yes (cloud or direct) |
| `03a_application_demo_replay.ipynb` | Six-solver pipeline comparison on bundled data. | No |
| `03b_application_demo_realtime.ipynb` | Same comparison live, on your graph. | Yes (cloud or direct) |
| `04_tutorial_end_to_end.ipynb` | Self-contained single-notebook walkthrough — classical CG, quantum CG, direct MILP on a synthetic antenna instance. | Optional (cloud default) |

**Recommended reading order for new colleagues:** `04` (single-notebook
end-to-end) → `01` (theory deep dive) → `02a / 02b` (algorithm step) →
`03a / 03b` (pipeline comparison).

---

## Slide decks (`slides/`)

| Deck | File | Audience | Pages |
|---|---|---|---|
| Deep dive | `slides/deep_dive/deep_dive.pdf` | Technical (engineering / research). Mirrors notebooks 01–03. | 24 |
| Tutorial | `slides/tutorial/tutorial.pdf` | Technical customer. Mirrors notebook 04. | 14 |
| Overview | `slides/overview/overview.pdf` | High-level customer / executive pitch. No math. | 9 |

All decks use the canonical QCi beamer template (Madrid theme,
QCi colors, Raleway font). Rebuild any deck with its local `build.sh`,
or `make -C notebooks/slides` for all three. See
`slides/README.md` for details.

---

## Backend modes

Every notebook that talks to Dirac exposes a single toggle:

```python
BACKEND = "replay"   # "replay" | "cloud" | "direct"
```

* **`replay`** — reads pre-recorded raw samples from
  `data_bundle/<iid>/<method>/raw_samples/call_*.pkl`. No credentials
  required. The bundled instance is `er_n20_p70_s0`.
* **`cloud`** — submits to the QCi cloud API via
  `eqc_models.solvers.Dirac3ContinuousCloudSolver`. Requires **both**
  `QCI_TOKEN` and `QCI_API_URL` (default `https://api.qci-prod.com`).
* **`direct`** — submits to on-prem Dirac-3 hardware via the gRPC
  `eqc-direct` package. Requires `EQC_DIRECT_IP_ADDRESS` (default
  `172.18.41.228`) and `EQC_DIRECT_PORT` (default `50051`).

### Credential resolution order

`_demo_utils.load_dotenv_if_present()` searches for `.env` files in this
order, never overwriting an already-set env var:

1. `./.env` (repo root)
2. `./quantum-branch-price/.env`
3. `~/Code/qci/skills/.env`
4. `~/Code/qci/skills/qci-eqc-models/.env`
5. `~/Code/qci/skills/max-clique-skills/.env`

If a variable is still missing, the notebook prompts interactively
(token via `getpass`, URL/IP via plain `input`).

### Optional: Hexaly ILP solver

Notebooks `03a`, `03b`, and `04` include **Hexaly** as a fast
commercial-grade direct ILP baseline alongside the always-available
HiGHS solver. Hexaly is licensed and not pip-installable; if it isn't
available the notebook cell prints a skip message and continues.

To enable Hexaly:

1. Download from [hexaly.com](https://www.hexaly.com/) (requires
   license) and install to `/opt/hexaly_14_5/` (or similar).
2. Set environment in each shell where you run the notebooks:
   ```bash
   export PYTHONPATH=/opt/hexaly_14_5/bin/python:$PYTHONPATH
   # macOS:
   export DYLD_LIBRARY_PATH=/opt/hexaly_14_5/bin:$DYLD_LIBRARY_PATH
   # Linux:
   # export LD_LIBRARY_PATH=/opt/hexaly_14_5/bin:$LD_LIBRARY_PATH
   ```
3. Place the license at `/opt/hexaly_14_5/license.dat`.
4. Launch `jupyter lab` from the same shell.

`pyproject.toml` carries a `hexaly` placeholder extra solely to record
the intent — there is no PyPI wheel to install.

---

## Saving live runs locally

Realtime notebooks (`02b`, `03b`, and `04` when not in replay) default
to `SAVE_RUN = True` and write each Dirac call to
`runs/<UTC-timestamp>_<label>/`:

```
runs/20260508T172500Z_cg_cloud/
├── graph.json
├── raw_samples/
│   ├── index.jsonl
│   └── call_00000.pkl ... call_NN.pkl
└── comparison.json     # (03b only) solver stats
```

The pickle schema is binary-compatible with `data_bundle/`, so the same
`ReplayDiracOracle` works on both. To replay a saved run offline:

```python
import _demo_utils as U
oracle = U.replay_oracle_from_run("runs/20260508T172500Z_cg_cloud")
```

`runs/` is gitignored.

---

## Bundled data (`data_bundle/`)

~220 KB of vendored replay data. The notebooks transparently prefer this
folder; if a developer has the full parent repo and `data_bundle/` is
absent, `_demo_utils.py` falls back to `../RF-branching/...`. See
`data_bundle/README.md` for the file-by-file provenance.

The bundled instance is **ER(20, 0.7, seed=0)** — 20 vertices, 131
edges, chromatic number 8. A full CG run replays in ~200 ms vs ~9
minutes on the original device.

---

## Direct-backend workarounds (May 2026)

Three issues in the upstream `quantum_colgen.pricing.dirac_oracle`
direct path are patched at notebook import time by `_demo_utils.py`:

1. **Wrong default IP**: skill docs say `172.18.41.79`, but that
   endpoint is offline; the working address is `172.18.41.228:50051`.
2. **Port type**: `_resolve_direct_config` casts to `int`, but
   `eqc_direct.client.EqcClient` requires `port` as a `str`.
3. **Lock-id discarded**: `_direct_solve_qp` calls `wait_for_lock()` but
   ignores the returned `(lock_id, t1, t2)` tuple, so
   `solve_sum_constrained` is invoked with the default empty `lock_id`
   and the device responds with `err_code=4 lock mismatch`. Patch
   captures and threads the lock_id.
4. **Wrong response key**: `_direct_extract_solutions` looks for
   `"solutions"` (plural) but `solve_sum_constrained` returns
   `"solution"` (singular). Patch checks both.

These patches live in `_install_direct_workarounds()` and run lazily the
first time `make_dirac_oracle("direct")` is called. They mutate the
imported `dirac_oracle` module in-place; if upstream is fixed, the
assignments become no-ops and can be removed.

---

## Verified behavior

* All five "core" notebooks (01, 02a, 03a, 02b/03b in replay) execute
  cleanly with `errors=0` via `jupyter nbconvert --execute`.
* `04_tutorial_end_to_end.ipynb` runs end-to-end against the QCi cloud
  API (defaults to `BACKEND="cloud"`), reproducing χ = 6 on a
  12-antenna synthetic subgraph in ~50 s total wallclock.
* Replay mode is genuinely offline — it monkey-patches both
  `_dirac_solve_qp` (used by `method="filter"`) and the
  `Dirac3ContinuousCloudSolver` class symbol (used directly by
  `method="gibbons"`).
