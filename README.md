# Quantum Column Generation for Graph Coloring

Implementation of the quantum column generation algorithm from
[arXiv:2301.02637v2](https://arxiv.org/abs/2301.02637) for minimum vertex
graph coloring, using QCi's Dirac-3 photonic quantum annealer as the
pricing oracle.

## Overview

Column generation decomposes graph coloring into:

1. **Restricted Master Problem (RMP)** — LP relaxation selecting which
   independent sets to use as color classes.
2. **Pricing Subproblem (PSP)** — find a maximum-weight independent set
   (MWIS) that could improve the LP bound.

The PSP is solved by a pluggable oracle:

| Oracle | Backend | Description |
|--------|---------|-------------|
| `classical` | `scipy.optimize.milp` | Exact MILP — baseline |
| `pulser` | Pulser + QuTiP | Neutral-atom quantum simulator |
| `dirac` | QCi Dirac-3 | Photonic quantum annealer (cloud) |

The Dirac oracle uses the **Gibbons weighted Motzkin-Straus** formulation:
it constructs a quadratic program on the complement graph with dual-variable
weights, submits it to Dirac-3, and extracts the MWIS from the continuous
solution support.

## Installation

```bash
uv venv --python 3.12 && uv sync --extra dev

# Editable reference dependency
uv pip install -e ./mis-spectral-graph-solver
```

**Note:** Environment is pinned to `numpy<2, networkx<3` for compatibility
with `eqc-models` and `qci-client`.

## Quick start

```bash
# Classical baseline on predefined test graphs
uv run python scripts/run_colgen.py --test --oracle classical -v

# Dirac oracle (requires QCI_TOKEN)
source ~/.zshrc
uv run python scripts/run_colgen.py --test --oracle dirac -v

# Random Erdos-Renyi graph
uv run python scripts/run_colgen.py --oracle classical --nodes 15 --edge-prob 0.4
```

## Tests

```bash
uv run pytest tests/unit/ -v          # 17 classical tests
uv run pytest tests/dirac/ -v         # Dirac tests (needs QCI_TOKEN)
```

## Benchmarks

```bash
# Full Dirac vs classical benchmark
uv run python -u scripts/benchmark.py --er-sizes 8 10 12 15 --er-probs 0.3 0.5 \
  --max-iterations 500 --json results/benchmark.json

# Validate coloring solutions (saves full color class assignments)
uv run python scripts/validate_results.py --oracle classical \
  --save-colorings results/colorings.json
```

### Benchmark Results

Benchmarks were run on predefined graphs with known chromatic numbers and
Erdos-Renyi random graphs G(n, p) with seed=42.

#### Known-chromatic-number graphs (n=3-6)

Both oracles find the exact chromatic number on all 6 predefined graphs:

| Graph | n | m | chi (classical) | chi (Dirac) | Expected |
|-------|---|---|-----------------|-------------|----------|
| paper_5vertex | 5 | 6 | 3 | 3 | 3 |
| triangle | 3 | 3 | 3 | 3 | 3 |
| complete_k4 | 4 | 6 | 4 | 4 | 4 |
| path_p4 | 4 | 3 | 2 | 2 | 2 |
| cycle_c5 | 5 | 5 | 3 | 3 | 3 |
| wheel_w5 | 6 | 10 | 4 | 4 | 4 |

#### Erdos-Renyi random graphs: Dirac vs Classical (n=15-30)

| Graph | n | m | Classical | Dirac | Winner | Greedy |
|-------|---|---|-----------|-------|--------|--------|
| ER(15,0.3) | 15 | 38 | 6 | **4** | Dirac | 4 |
| ER(15,0.5) | 15 | 52 | 6 | 6 | Tie | 6 |
| ER(20,0.3) | 20 | 67 | 6 | 6 | Tie | 5 |
| ER(20,0.5) | 20 | 98 | **7** | 8 | Classical | 7 |
| ER(25,0.3) | 25 | 105 | 7 | 7 | Tie | 6 |
| ER(25,0.5) | 25 | 151 | 8 | 8 | Tie | 8 |
| ER(30,0.3) | 30 | 143 | 9 | **8** | Dirac | 6 |
| ER(30,0.5) | 30 | 217 | **8** | 9 | Classical | 7 |

**Score: Dirac 2, Classical 2, Tie 4.** Neither consistently dominates.
Dirac found the optimal chi=4 for ER(15,0.3) where classical CG produced
chi=6.

#### Classical CG scaling (n=15-100, max_iterations=500)

| Graph | n | m | CG chi | Greedy chi | Iterations | Time (s) |
|-------|---|---|--------|------------|------------|----------|
| ER(15,0.3) | 15 | 38 | 6 | 4 | 18 | 0.02 |
| ER(20,0.3) | 20 | 67 | 6 | 5 | 35 | 0.07 |
| ER(30,0.3) | 30 | 143 | 9 | 6 | 48 | 0.25 |
| ER(40,0.3) | 40 | 244 | 8 | 8 | 130 | 4.89 |
| ER(50,0.3) | 50 | 368 | 11 | 8 | 189 | 16.13 |
| ER(75,0.3) | 75 | 804 | 17 | 10 | 318 | 217 |
| ER(100,0.3) | 100 | 1477 | 26 | 12 | 432 | 1066 |

CG converges in the LP sense but the final ILP has a significant integrality
gap on large sparse graphs. Branch-and-price would be needed to close this gap.

#### Validation

All coloring solutions pass detailed validation (0 edge violations, full
vertex coverage, no duplicates) for both classical and Dirac oracles.

```bash
uv run python scripts/validate_results.py --oracle classical --save-colorings results/colorings.json
uv run python scripts/validate_results.py --summary results/colorings.json
```

## Project structure

```
src/quantum_colgen/
    column_generation.py       # CG loop + verify_coloring + validate_coloring
    master_problem.py          # RMP (scipy.optimize.linprog) + ILP (milp)
    graphs.py                  # Test graph generators with known chi
    pricing/
        base.py                # Abstract PricingOracle interface
        classical.py           # Exact MILP-based MWIS
        pulser_oracle.py       # Pulser neutral-atom MWIS
        dirac_oracle.py        # QCi Dirac-3 MWIS (gibbons + filter)
scripts/
    run_colgen.py              # CLI entry point
    benchmark.py               # Benchmark suite with comparison tables
    validate_results.py        # Coloring validation + solution export
tests/
    unit/                      # 17 classical CG tests
    dirac/                     # Dirac oracle tests (needs QCI_TOKEN)
    pulser/                    # Pulser oracle tests
results/                       # Benchmark JSON output
```

## References

- Botter, Boman, et al. "Quantum Column Generation." arXiv:2301.02637v2 (2023).
- Motzkin, Straus. "Maxima for graphs and a new proof of a theorem of Turan." *Canadian J. Math.* (1965).
- Gibbons, Aaronson, Bollobas. "Characterising the maximum weight independent set with a weighted generalisation of the Motzkin-Straus theorem." (2015).
