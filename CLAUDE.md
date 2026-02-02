# CLAUDE.md — quantum-column-generation

## What is this repo?

Reproduces the quantum column generation algorithm from **arXiv:2301.02637v2** for
minimum vertex graph coloring. Supports three pricing oracles: classical MILP,
Pulser neutral-atom simulator, and QCi Dirac-3 quantum annealer.

## Build & environment

Pinned to eqc-models-compatible versions (numpy<2, networkx<3).

```bash
uv venv --python 3.12 && uv sync --extra dev
uv pip install -e ./mis-spectral-graph-solver              # motzkinstraus (editable)
# Pulser oracle (conflicts with eqc-models — use a separate venv):
# uv pip install pulser maximum-independent-set
```

## Run tests

```bash
uv run pytest tests/unit/ -v                # Classical CG — always works
source ~/.zshrc                             # Loads QCI_TOKEN
uv run pytest tests/dirac/ -v              # Requires QCI_TOKEN
uv run pytest tests/pulser/ -v              # Requires separate pulser venv
```

## Run CLI

```bash
uv run python scripts/run_colgen.py --test --oracle classical --verbose
uv run python scripts/run_colgen.py --test --oracle dirac --verbose
uv run python scripts/run_colgen.py --oracle classical --nodes 12 --edge-prob 0.5
```

## Run benchmarks

```bash
# Full Dirac vs classical benchmark
source ~/.zshrc
uv run python -u scripts/benchmark.py --er-sizes 8 10 12 15 --er-probs 0.3 0.5 \
  --max-iterations 500 --json results/benchmark.json

# Classical only (no Dirac API calls)
uv run python -u scripts/benchmark.py --classical-only \
  --er-sizes 8 10 12 15 20 --er-probs 0.3 0.5 --max-iterations 500

# Validate coloring solutions
uv run python scripts/validate_results.py --oracle classical --save-colorings results/colorings.json
uv run python scripts/validate_results.py --summary results/colorings.json
```

Results are saved to `results/` as JSON files.

## Project layout

```
src/quantum_colgen/
    __init__.py
    column_generation.py       # CG loop orchestrator + validate_coloring()
    master_problem.py          # RMP (LP relaxation) + final ILP
    graphs.py                  # Test graph generators
    pricing/
        base.py                # Abstract PricingOracle
        classical.py           # Exact MILP MWIS
        pulser_oracle.py       # Neutral-atom (Pulser) MWIS
        dirac_oracle.py        # QCi Dirac-3 MWIS (filter + gibbons methods)
tests/
    unit/                      # Classical-only tests (17 tests)
    pulser/                    # @pytest.mark.pulser
    dirac/                     # @pytest.mark.dirac (needs QCI_TOKEN)
scripts/
    run_colgen.py              # CLI entry point
    benchmark.py               # Dirac vs classical benchmark
    validate_results.py        # Coloring validation + solution export
results/                       # Benchmark JSON output
notebooks/                     # Tutorial notebook
mis-spectral-graph-solver/     # Editable dep (excluded from git)
```

## Key conventions

- Node labels are always integers 0..n-1.
- Pricing oracles implement `PricingOracle.solve(graph, dual_vars) -> List[Set[int]]`.
- A column is profitable when the sum of dual weights exceeds 1.
- `pulser` and `eqc-models` have conflicting transitive deps and cannot coexist.
  Use separate venvs or install only one at a time.
- Environment is pinned to `numpy<2, networkx<3` for eqc-models compatibility.
- Default `max_iterations=500` in column_generation(); 50 is insufficient for n>25.

## Dependencies on mis-spectral-graph-solver

The Dirac oracle in this repo is self-contained. The `motzkinstraus` package from
`mis-spectral-graph-solver/` is installed as an editable dep for reference and for
running the original scripts, but `quantum_colgen` does not import from it.
