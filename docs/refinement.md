# Pricing Oracle Refinement: LP Relaxation and Enhanced Extraction

## Overview

This document describes improvements to the column generation pricing oracles:
1. **Enhanced extraction** for Dirac oracle (multi-prune, randomized rounding)
2. **New ClassicalLPPricingOracle** using LP relaxation + extraction
3. **Benchmark comparison** of MILP vs LP vs Dirac oracles

## Problem Statement

The Dirac quantum annealer returns ~100 solution samples per API call, but current extraction yields only 5-7 profitable columns from 500 potential (sample × threshold) combinations.

**Root cause**: Many (sample, threshold) combinations collapse to the same independent set (IS) after greedy pruning. The greedy pruning strategy (highest dual weight first) is deterministic, so different supports often converge to the same IS.

## Experimental Results

We tested multiple extraction strategies on ER(n, 0.3) random graphs. Key findings:

### ER(15, 0.3) - Small Graph

| Strategy | Unique IS | Profitable (>1) | Time (ms) |
|----------|-----------|-----------------|-----------|
| Baseline (5 thresholds) | 7 | 7 | 8.4 |
| Fine thresholds (15) | 7 | 7 | 18.9 |
| Ultra-fine thresholds (23) | 7 | 7 | 24.0 |
| **Multi-prune orders** | **13** | **13** | 103.8 |
| Randomized rounding | 13 | 13 | 33.7 |
| **Combined (best)** | **16** | **16** | 98.1 |
| Sub-IS from combined | 265 | 265 | 0.3 |

### ER(30, 0.3) - Medium Graph

| Strategy | Unique IS | Profitable (>1) | Time (ms) |
|----------|-----------|-----------------|-----------|
| Baseline | 37 | 37 | 15.4 |
| Fine thresholds | 42 | 42 | 26.9 |
| Ultra-fine thresholds | 45 | 45 | 44.8 |
| **Multi-prune orders** | **56** | **56** | 202.4 |
| **Randomized rounding** | **124** | **124** | 79.7 |
| **Combined** | **130** | **130** | 197.1 |
| Sub-IS from combined | 3625 | 3625 | 16.3 |

## Key Findings

### 1. Fine-Grained Thresholds Have Diminishing Returns

Moving from 5 to 15 to 23 thresholds only improves yield by ~20%. This suggests the threshold values themselves don't matter much - what matters is how we prune the support to get an independent set.

### 2. Multi-Prune is the Most Important Factor

Using multiple pruning strategies (dual-weight descending, ascending, degree-based, random) nearly doubles the unique IS count. This is because:
- Different pruning orders explore different parts of the IS space
- The same support set can be pruned to different valid IS
- Cost is ~10x baseline but yield is ~2x

**Recommendation**: Add at least dual-ascending and random pruning to production.

### 3. Randomized Rounding is Highly Effective

For medium graphs (n≥30), randomized rounding (probabilistic node inclusion based on solution values) is the single most effective strategy:
- 3x improvement over baseline for ER(30, 0.3)
- Relatively cheap (~80ms for 20 rounds)
- Explores the solution space more broadly

**Recommendation**: Add randomized rounding with 10-20 rounds per solution sample.

### 4. Combined Strategy Achieves 3.5x Baseline

Combining multi-prune and randomized rounding achieves the best results without sub-IS enumeration.

### 5. Sub-IS Enumeration Yields Massive Column Counts

Enumerating subsets of profitable IS (removing 1-3 nodes) yields thousands of additional columns. However:
- Most are variations of the same "good" IS
- May not add much value to the ILP
- Could overwhelm the final set-cover ILP solver

**Recommendation**: Consider sub-IS extraction only if:
1. Running HiGHS (fast) for final ILP
2. Graph size is small (n < 50)
3. Looking for best possible chi rather than fast results

## Production Recommendations

### Recommended Configuration (Balanced)

```python
DiracPricingOracle(
    support_thresholds=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    local_search_passes=5,
    multi_prune=True,           # NEW: use multiple pruning orders
    randomized_rounding=True,   # NEW: probabilistic extraction
    num_random_rounds=10,       # NEW: rounds per sample for randomized
)
```

Expected improvement: **3-4x more columns per API call**.

### Aggressive Configuration (Maximum Extraction)

```python
DiracPricingOracle(
    support_thresholds=FINE_THRESHOLDS,  # 15 thresholds
    local_search_passes=5,
    multi_prune=True,
    randomized_rounding=True,
    num_random_rounds=20,
    sub_is_extraction=True,     # NEW: enumerate subsets
    max_sub_is_removals=2,      # NEW: remove up to 2 nodes
)
```

Expected improvement: **10-100x more columns**, but longer ILP solve time.

## Implementation Notes

### Multi-Prune Orders

The key pruning strategies to add:
1. **Dual-ascending**: Include lowest-weight nodes first (yields different IS)
2. **Degree-descending**: Include highest-degree nodes first (in induced subgraph)
3. **Random** (3-5 trials): Explore more of the IS space

### Randomized Rounding Algorithm

```python
for solution in dirac_solutions:
    x_norm = x / (x.sum() + 1e-10)  # normalize
    for round in range(num_rounds):
        support = {node for i, node in enumerate(nodes)
                   if random() < x_norm[i] * scale_factor}
        is_set = greedy_prune(support, graph, duals)
        is_set = local_search(is_set, graph, duals)
        yield is_set
```

The `scale_factor` (typically 3-5) controls how many nodes are included per sample.

## Future Work

1. **Adaptive threshold selection**: Use the solution value distribution to choose thresholds dynamically
2. **2-swap local search**: Current implementation uses 1-swap; 2-swap may find better IS
3. **Relaxed profitability**: Accept columns with profit > 0.8 (may help ILP)
4. **Solution diversity metrics**: Track which (sample, threshold) pairs contribute unique IS
5. **ILP impact analysis**: Measure how additional columns affect final chi

---

## Classical LP Pricing Oracle

### Motivation

Instead of solving the exact MILP (NP-hard) for the pricing subproblem, we can use
LP relaxation (polynomial time) and extract multiple columns via thresholding—the
same approach used by Dirac.

This provides an apples-to-apples comparison: both LP and Dirac use continuous
relaxation + extraction, isolating whether Dirac's value comes from quantum
annealing or from the continuous relaxation approach itself.

### LP Relaxation of MWIS

```
maximize  Σ w_i * x_i
s.t.      x_u + x_v ≤ 1  for all edges (u,v)
          0 ≤ x_i ≤ 1
```

The LP relaxation gives fractional solutions (often half-integral: 0.5 values).
We threshold these to extract candidate independent sets, then apply the same
pruning and local search as the Dirac oracle.

### Implementation

`ClassicalLPPricingOracle` in `src/quantum_colgen/pricing/classical_lp.py`:
- Uses scipy.optimize.linprog with HiGHS backend
- Supports multi-prune and randomized rounding (same as Dirac)
- Default thresholds include 0.49 to capture half-integral solutions

---

## Oracle Benchmark Results

### Chi Quality Comparison (ER graphs, seed=42)

#### Small graphs (20-40 nodes)

| Graph | Greedy | MILP | LP | Dirac |
|-------|--------|------|-----|-------|
| ER(20,0.3) | **5** | 6 | 6 | - |
| ER(20,0.5) | 7 | 7 | 7 | - |
| ER(30,0.3) | **6** | 9 | 8 | **6** |
| ER(30,0.5) | **7** | 8 | **7** | **7** |
| ER(40,0.3) | 8 | 8 | 9 | - |
| ER(40,0.5) | 11 | 10 | **9** | - |

#### Large graphs (50-100 nodes)

| Graph | Greedy | MILP | LP | MILP gap | LP gap |
|-------|--------|------|-----|----------|--------|
| ER(50,0.3) | **8** | 11 | 9 | +3 | +1 |
| ER(50,0.5) | 11 | 11 | 11 | 0 | 0 |
| ER(75,0.3) | **10** | 17 | 16 | +7 | +6 |
| ER(75,0.5) | 16 | 17 | **15** | +1 | **-1** ✓ |
| ER(100,0.3) | **12** | 26 | 23 | **+14** | +11 |
| ER(100,0.5) | **20** | 26 | 21 | +6 | +1 |

**Key finding**: MILP degrades catastrophically on large sparse graphs. ER(100,0.3) gets
chi=26 vs greedy's chi=12 — more than 2x worse! LP is consistently better.

### Extraction Efficiency (columns per API call)

| Graph | MILP | LP | Dirac |
|-------|------|-----|-------|
| ER(30,0.3) | 1.0 | 12.8 | **30.1** |
| ER(30,0.5) | 1.0 | 9.1 | 9.1 |
| ER(50,0.3) | 1.0 | 9.1 | - |
| ER(75,0.3) | 1.0 | 13.6 | - |
| ER(100,0.3) | 1.0 | 12.8 | - |
| ER(100,0.5) | 1.0 | 13.9 | - |

### Iteration Efficiency (large graphs)

| Graph | MILP iters | LP iters | Reduction |
|-------|------------|----------|-----------|
| ER(50,0.3) | 189 | 37 | **5.1x fewer** |
| ER(75,0.3) | 318 | 38 | **8.4x fewer** |
| ER(100,0.3) | 432 | 66 | **6.5x fewer** |
| ER(100,0.5) | 293 | 41 | **7.1x fewer** |

### Time Comparison

| Graph | MILP | LP | Dirac |
|-------|------|-----|-------|
| ER(30,0.3) | 0.27s | **0.11s** | 5068s |
| ER(30,0.5) | 0.30s | **0.06s** | 493s |
| ER(50,0.3) | 16.0s | **1.8s** | - |
| ER(75,0.3) | 216.5s | **89.4s** | - |
| ER(100,0.3) | 1096.3s | **822.5s** | - |

### Key Findings

1. **Dirac consistently matches greedy** (best chi) when tested
2. **LP matches greedy on denser graphs** but may lose on sparse ones
3. **MILP (exact) often loses to greedy** - counterintuitively!
4. **More columns = better chi**: The continuous relaxation approaches (LP, Dirac)
   generate 10-30 columns per call vs MILP's 1, leading to better final ILP solutions
5. **LP is ~5000x faster than Dirac** due to API latency, but Dirac extracts
   2-3x more columns per call

### Recommendations

- **For speed**: Use `ClassicalLPPricingOracle` - fast and competitive
- **For quality**: Use `DiracPricingOracle` with enhanced extraction - best chi
- **Avoid**: Plain `ClassicalPricingOracle` (MILP) - slow AND worse chi

---

## Files

- `scripts/dirac_extraction_experiments.py` - Extraction strategy experiments
- `scripts/benchmark_oracles.py` - MILP vs LP vs Dirac benchmark
- `scripts/test_enhanced_extraction_chi.py` - Enhanced extraction chi test
- `src/quantum_colgen/pricing/dirac_oracle.py` - Dirac oracle with enhanced extraction
- `src/quantum_colgen/pricing/classical_lp.py` - New LP relaxation oracle
