# Benchmark Results

Organized benchmark results for quantum column generation graph coloring experiments.

## Folder Structure

```
benchmarks/
├── README.md                      # This file
├── er40_0.3/                      # ER(40, 0.3) graph results
│   ├── graph.json                 # Graph structure
│   ├── greedy.json                # Greedy coloring result
│   ├── lp.json                    # LP CG result
│   ├── dirac.json                 # Dirac CG result
│   ├── exact_ilp_highs.json       # Direct ILP result (HiGHS)
│   ├── exact_ilp_hexaly.json      # Direct ILP result (Hexaly)
│   └── summary.json               # Combined summary
├── er50_0.3/
│   └── (same structure)
├── er50_0.5/
│   └── (same structure)
├── er50_0.7/
│   └── (same structure)
├── er50_0.9/
│   └── (same structure)
├── er75_0.3/
│   └── (same structure)
├── er100_0.3/
│   └── (same structure)
└── slides_data.json               # Aggregated data for manim slides
```

## Data Formats

### graph.json

```json
{
  "graph_name": "ER(40,0.3)",
  "n_nodes": 40,
  "n_edges": 244,
  "seed": 42,
  "edges": [[0, 2], [0, 3], ...]
}
```

### Method results (greedy.json, lp.json, dirac.json)

```json
{
  "method": "dirac",
  "graph_name": "ER(40,0.3)",
  "n_nodes": 40,
  "n_edges": 244,
  "chi": 7,
  "color_classes": [[...], [...], ...],
  "valid": true,
  "iterations": 11,
  "columns_found": 291,
  "wall_seconds": 678.87,
  "timing": {
    "num_api_calls": 12,
    "total_api_seconds": 670.0,
    "total_extract_seconds": 2.4,
    "avg_columns_per_call": 24.25
  },
  "oracle_config": {
    "method": "gibbons",
    "num_samples": 100,
    "multi_prune": true,
    "randomized_rounding": true,
    "num_random_rounds": 10,
    "random_seed": 42
  },
  "timestamp": "2026-02-05T11:00:00Z"
}
```

### summary.json

```json
{
  "graph_name": "ER(40,0.3)",
  "n_nodes": 40,
  "n_edges": 244,
  "methods": {
    "greedy": {"chi": 8, "wall_seconds": 0.0},
    "lp": {"chi": 8, "iterations": 25, "wall_seconds": 0.77},
    "dirac": {"chi": 7, "iterations": 11, "wall_seconds": 678.87}
  },
  "best_chi": 7,
  "winner": "dirac"
}
```

## Reproducing Results

### Run all benchmarks

```bash
# Classical only (greedy + LP)
uv run python scripts/run_benchmark_full.py --graphs 40 50 75 100 --classical-only

# With Dirac (requires QCI_TOKEN)
source ~/.zshrc
uv run python scripts/run_benchmark_full.py --graphs 40 50 75 100
```

### Run specific graph

```bash
uv run python scripts/run_benchmark_full.py --graphs 75 --edge-prob 0.3
```

### Validate colorings

```bash
uv run python scripts/validate_results.py --from-json benchmarks/slides_data.json
```

## Key Parameters

All benchmarks use consistent parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `seed` | 42 | Random seed for graph generation |
| `edge_prob` | 0.3 | Erdos-Renyi edge probability |
| `max_iterations` | 500 | CG iteration limit |
| `num_samples` | 100 | Dirac samples per API call |
| `multi_prune` | true | Enable multi-pruning extraction |
| `randomized_rounding` | true | Enable randomized rounding |

## Results Summary

### Complete Benchmark Results (300s timeout for ILP)

| Graph | Nodes | Edges | Greedy | LP CG | Dirac CG | HiGHS | Hexaly | Best χ | Winner |
|-------|-------|-------|--------|-------|----------|-------|--------|--------|--------|
| ER(40,0.3) | 40 | 244 | 8 | 8 | 7 | **6** ✓ | **6** ✓ | 6 | ILP |
| ER(50,0.3) | 50 | 368 | 8 | 10 | 8 | **6** ✓ | **6** ✓ | 6 | ILP |
| ER(50,0.5) | 50 | 596 | 11 | 11 | 10 | **9** | **9** | 9 | ILP |
| ER(50,0.7) | 50 | 856 | 17 | 14 | 14 | 14 | **13** | 13 | Hexaly |
| ER(50,0.9) | 50 | 1104 | 26 | - | 23 | 23 | 23 | 23 | Tie |
| ER(75,0.3) | 75 | 804 | 10 | 17 | 14 | **8** | **8** | 8 | ILP |
| ER(100,0.3) | 100 | 1477 | 12 | 22 | 19 | **11** | **11** | 11 | ILP |

✓ = proven optimal (0% MIP gap)

### Key Findings

1. **Exact ILP dominates**: With 300s timeout, HiGHS/Hexaly find better solutions than all other methods
2. **Sparse graphs (p=0.3)**: ILP proves optimality for 40-50 nodes, finds best solutions for 75-100 nodes
3. **Dense graphs**: ILP still competitive but harder to prove optimality (higher MIP gaps)
4. **Hexaly vs HiGHS**: Similar performance; Hexaly found χ=13 for ER(50,0.7), best known
5. **Column generation value**: CG methods (Dirac, LP) are faster but find suboptimal solutions
6. **Dirac vs Greedy**: Dirac beats greedy on small dense graphs but loses on larger sparse graphs

### MIP Gaps (300s timeout)

| Graph | HiGHS χ | Gap | Hexaly χ | Gap | Lower Bound |
|-------|---------|-----|----------|-----|-------------|
| ER(40,0.3) | 6 | 0% | 6 | 0% | 6 |
| ER(50,0.3) | 6 | 0% | 6 | 0% | 6 |
| ER(50,0.5) | 9 | 22% | 9 | - | 7 |
| ER(50,0.7) | 14 | 29% | 13 | - | 10 |
| ER(75,0.3) | 8 | 38% | 8 | 38% | 5 |
| ER(100,0.3) | 11 | 45% | 11 | 45% | 6 |

## Using Results in Manim Slides

The `slides_data.json` file is consumed by the manim presentation:

```python
# manim/src/quantum_colgen_slides/data/benchmark_data.py
# Update this file or copy slides_data.json to:
# manim/src/quantum_colgen_slides/data/graph_colorings.json
```
