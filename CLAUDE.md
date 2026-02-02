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
# Full Dirac vs classical benchmark (original, small graphs)
source ~/.zshrc
uv run python -u scripts/benchmark.py --er-sizes 8 10 12 15 --er-probs 0.3 0.5 \
  --max-iterations 500 --json results/benchmark.json

# Improved benchmark with timing breakdown (classical + greedy)
uv run python -u scripts/benchmark_improved.py --classical-only \
  --er-sizes 30 50 75 100 --er-probs 0.3 0.5 --max-iterations 500 \
  --json results/benchmark_improved_classical.json

# Improved benchmark with Dirac + classical + greedy
source ~/.zshrc
uv run python -u scripts/benchmark_improved.py --dirac \
  --er-sizes 30 --er-probs 0.3 0.5 --num-samples 50 --max-iterations 500 \
  --json results/benchmark_improved_dirac.json

# Classical only (no Dirac API calls, original script)
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
    timing.py                  # OracleTimer for per-call timing instrumentation
    pricing/
        base.py                # Abstract PricingOracle
        classical.py           # Exact MILP MWIS (instrumented with OracleTimer)
        pulser_oracle.py       # Neutral-atom (Pulser) MWIS
        dirac_oracle.py        # QCi Dirac-3 MWIS — multi-sample, multi-threshold,
                               #   local search, returns multiple columns per call
tests/
    unit/                      # Classical-only tests (30 tests)
    pulser/                    # @pytest.mark.pulser
    dirac/                     # @pytest.mark.dirac (needs QCI_TOKEN, 7 tests)
scripts/
    run_colgen.py              # CLI entry point
    benchmark.py               # Dirac vs classical benchmark (original)
    benchmark_improved.py      # Improved benchmark: CG vs greedy with timing breakdown
    validate_results.py        # Coloring validation + solution export
results/                       # Benchmark JSON output
notebooks/                     # Tutorial notebook
mis-spectral-graph-solver/     # Editable dep (excluded from git)
```

## Key conventions

- Node labels are always integers 0..n-1.
- Pricing oracles implement `PricingOracle.solve(graph, dual_vars) -> List[Set[int]]`.
- A column is profitable when the sum of dual weights exceeds 1.
- Oracles expose `.timer` (an `OracleTimer`) for per-call timing instrumentation.
- `pulser` and `eqc-models` have conflicting transitive deps and cannot coexist.
  Use separate venvs or install only one at a time.
- Environment is pinned to `numpy<2, networkx<3` for eqc-models compatibility.
- Default `max_iterations=500` in column_generation(); 50 is insufficient for n>25.

## Dirac oracle improvements

The Dirac pricing oracle uses several techniques to maximize column yield per API call:

- **Multi-sample extraction**: processes ALL solution vectors from Dirac (not just the first).
- **Multi-threshold support extraction**: tries thresholds `[0.005, 0.01, 0.05, 0.1, 0.2]`
  on each solution vector — different thresholds yield different IS after pruning.
- **1-swap local search**: refines each extracted IS via greedy improvement (configurable
  via `local_search_passes`, default 5).
- **Multiple columns per call**: returns all unique profitable IS found across all
  (sample, threshold) combinations, typically 5-7 columns per Dirac API call.
- **Backward compatibility**: `support_threshold` (singular) still accepted.

## Git conventions

- Do NOT include `Co-Authored-By` lines in commit messages.

## Dependencies on mis-spectral-graph-solver

The Dirac oracle in this repo is self-contained. The `motzkinstraus` package from
`mis-spectral-graph-solver/` is installed as an editable dep for reference and for
running the original scripts, but `quantum_colgen` does not import from it.
