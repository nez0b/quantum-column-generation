# `notebooks/` — Quantum Column Generation tutorial series

Five notebooks walk through the algorithm from the underlying Motzkin-Straus
QP up to a full industry-style pipeline comparison.

| Notebook | Topic | API access? |
|---|---|---|
| `01_motzkin_straus.ipynb` | The QP at the heart of every Dirac call. SLSQP vs Dirac on a small graph. | Optional (replay default) |
| `02a_column_generation_replay.ipynb` | CG step-by-step on bundled ER(20, 0.7) data. | None |
| `02b_column_generation_realtime.ipynb` | Same content on **your** graph + live Dirac. Saves run for offline replay. | Yes (cloud or direct) |
| `03a_application_demo_replay.ipynb` | Six-solver pipeline comparison on bundled data. | None (uses replay) |
| `03b_application_demo_realtime.ipynb` | Same comparison live, on your graph. | Yes (cloud or direct) |

## Backend modes

Every notebook that talks to Dirac exposes a single toggle:

```python
BACKEND = "replay"   # "replay" | "cloud" | "direct"
```

* **`replay`** — reads pre-recorded raw samples from
  `RF-branching/instances/<iid>/<method>/raw_samples/call_*.pkl`. No
  credentials required. The bundled instance is `er_n20_p70_s0` (20-vertex
  ER graph, edge probability 0.7, seed 0).
* **`cloud`** — submits to the QCi cloud API via
  `eqc_models.solvers.Dirac3ContinuousCloudSolver`. Requires **both**
  `QCI_TOKEN` and `QCI_API_URL` (default `https://api.qci-prod.com`).
* **`direct`** — submits to on-prem Dirac-3 hardware via the gRPC
  `eqc-direct` package, which the project wraps inside
  `quantum_colgen.pricing.dirac_oracle.DiracPricingOracle(backend="direct")`.
  Requires `EQC_DIRECT_IP_ADDRESS` (skill default `172.18.41.79`) and
  `EQC_DIRECT_PORT` (default `50051`); `EQC_DIRECT_CERT_FILE` is optional.

### Credential resolution order

`_demo_utils.load_dotenv_if_present()` is called by every realtime
notebook before any Dirac instantiation. It looks for `.env` files in
this order, never overwriting an already-set env var:

1. `./.env` (repo root)
2. `./quantum-branch-price/.env`
3. `~/Code/qci/skills/.env`
4. `~/Code/qci/skills/qci-eqc-models/.env`
5. `~/Code/qci/skills/max-clique-skills/.env`

For testing, the bundled QCi skills carry working cloud creds at
`~/Code/qci/skills/qci-eqc-models/.env` (`QCI_API_URL` +
`QCI_TOKEN`). If a needed variable is still missing after env loading,
the notebook prompts the user interactively (token via `getpass`,
URL/IP via plain `input`).

## Saving live runs locally

Realtime notebooks (`02b`, `03b`) default to `SAVE_RUN = True` and write
each Dirac call to `notebooks/runs/<UTC-timestamp>_<label>/`:

```
notebooks/runs/20260508T172500Z_cg_cloud/
├── graph.json                         # instance metadata
├── raw_samples/
│   ├── index.jsonl                    # one entry per Dirac call
│   └── call_00000.pkl ... call_NN.pkl # raw solution vectors
└── comparison.json                    # (03b only) solver stats table
```

The pickle schema is binary-compatible with `RF-branching/instances/`, so
the same `ReplayDiracOracle` works on both. To rerun a saved run offline:

```python
import _demo_utils as U
oracle = U.replay_oracle_from_run("notebooks/runs/20260508T172500Z_cg_cloud")
```

`notebooks/runs/` is gitignored.

## Running the notebooks

`notebooks/` is a self-contained `uv` project with its own `pyproject.toml`
that pulls the parent package as an editable dependency. **Recommended**:

```bash
cd notebooks
uv venv             # creates notebooks/.venv (Python 3.12 by default)
uv sync             # cloud-only deps (jupyter + eqc-models + quantum-colgen)
uv sync --extra direct      # add eqc-direct for on-prem mode
uv run jupyter lab .         # launch
```

Or, from the repo root, share the parent venv:

```bash
uv run jupyter lab notebooks/
```

Both work because `_demo_utils.py` resolves paths from `__file__` and
adds `../src` to `sys.path` if the editable install isn't present.

The notebooks share a single Python module, `notebooks/_demo_utils.py`,
which provides path resolution, visualization helpers, the
`make_dirac_oracle(mode=...)` factory, the `CapturingPricingOracle`
wrapper, the `ReplayDiracOracle` shim that monkey-patches the cloud
solver inside `quantum_colgen.pricing.dirac_oracle`, and three
**direct-backend bug workarounds** (see below).

### Direct-backend workarounds (May 2026)

Three issues in the upstream `quantum_colgen.pricing.dirac_oracle`
direct path are patched at notebook import time by `_demo_utils.py`:

1. **Wrong default IP**: skill docs say `172.18.41.79`, but that endpoint
   is offline; current production hardware is `172.18.41.228:50051`.
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
imported `dirac_oracle` module in-place; if the upstream package is
fixed, the patches become no-ops on the assignment line and can be
removed.

## Bundled data

The `01`/`02a`/`03a` notebooks read these read-only artifacts (already
committed under `RF-branching/`):

* `RF-branching/slides/qcg_vs_cg_demo/data/psp_{01,02}.json` — per-PSP
  bundles with graph topology, dual weights, Dirac columns, LP columns,
  and Kamada-Kawai layouts.
* `RF-branching/instances/er_n20_p70_s0/{cg,qcg,bp_isf,...}/raw_samples/`
  — pickled raw Dirac samples for the bundled instance, used by
  `ReplayDiracOracle` to replay full CG runs offline.

## Verified replay behaviour

Replay mode is genuinely offline — it monkey-patches both
`_dirac_solve_qp` (used by `method="filter"`) and the
`Dirac3ContinuousCloudSolver` class symbol (used directly by
`method="gibbons"`). On the bundled ER(20, 0.7) instance a full CG run
replays in ~200 ms vs ~9 minutes on the original device.
