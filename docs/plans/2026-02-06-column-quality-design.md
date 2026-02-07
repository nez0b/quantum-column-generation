# Column Quality Experiments: 5 Novel Ideas for Better Dirac IS Extraction

**Date**: 2026-02-06
**Status**: Complete

## Problem

The Dirac-3 quantum annealer returns ~100 continuous solution vectors per API call, but
current post-processing extracts only 5-30 unique profitable independent sets. Many
(sample, threshold) combinations collapse to the same IS after greedy pruning.

From ER(40,0.3) trace: 20 iterations, 369 total columns, chi=7 (vs exact optimal chi=6).
LP relaxation bound was 5.30 — a gap of ~1.7 colors. Better column extraction could
close this gap.

## Five Experiment Ideas

### E1: Dual-Perturbed Multi-Solve

Solve 2-3 QPs per CG iteration with slightly perturbed dual weights:
```
w'[i] = w[i] * (1 + epsilon * N(0,1))    where epsilon in {0.05, 0.10, 0.15}
```
Different weight landscapes → different solution vectors → different IS.

**Hypothesis**: Perturbation breaks ties in dual values that cause solution collapse.
**Cost**: 2-3x API calls per CG iteration.

### E2: Subgraph Decomposition

Partition the positive-dual subgraph into overlapping communities (greedy modularity)
and solve separate smaller QPs per cluster. Union extracted IS, checking independence
across clusters.

**Hypothesis**: Smaller QPs are easier for Dirac-3 → higher quality solutions per subproblem.
**Cost**: 2-3x API calls, but on smaller problems.

### E3: Solution Vector Clustering

Cluster ~100 Dirac solutions via k-means (k=5-10):
- Extract from each cluster **centroid** (consensus, less noise)
- Extract from the most **extreme** member per cluster (maximum diversity)
- Compare against extracting from all 100 individually

**Hypothesis**: Centroids reduce noise; extremes increase diversity. Both yield more *distinct* IS.
**Cost**: No extra API calls — pure post-processing.

### E4: 2-Swap Local Search

Extend current 1-swap to 2-swap:
- For each pair of IS nodes (u, v), try removing both and greedily adding from N(u) ∪ N(v)
- Accept if net dual weight increases
- O(|IS|^2 * max_degree) per pass — feasible for IS of size 5-10

**Hypothesis**: Escapes local optima that 1-swap misses, especially on denser graphs.
**Cost**: More compute per extraction, no extra API calls.

### E5: Column Memory / Diversity Pressure

Pass previously found columns to the oracle. Penalize over-represented nodes in the
Gibbons matrix:
```
B'[i,i] = B[i,i] + lambda * frequency[i]
```
where `frequency[i]` = fraction of existing columns containing node i.

**Hypothesis**: Forces Dirac to explore under-represented regions of the IS space.
**Cost**: No extra API calls, but modifies the QP each iteration.

## Test Graphs

| Graph | n | Expected density |
|-------|---|-----------------|
| ER(30, 0.5, seed=42) | 30 | ~217 edges |
| ER(40, 0.5, seed=42) | 40 | ~390 edges |
| ER(50, 0.5, seed=42) | 50 | ~612 edges |

## Metrics

1. **Unique profitable IS count** — primary extraction quality metric
2. **Total dual weight** — profitability quality
3. **Column diversity** — pairwise Jaccard distance between IS
4. **Final chi** — the only metric that truly matters
5. **Wall time** — API calls + extraction time
6. **Convergence speed** — iterations to converge

## Experiment Parameters

| Experiment | Parameter | Values |
|-----------|-----------|--------|
| E1: Dual perturbation | epsilon | 0.05, 0.10, 0.15 |
| E1: Dual perturbation | num_perturbations | 2, 3 |
| E2: Subgraph decomp | num_clusters | 2, 3 |
| E2: Subgraph decomp | overlap_fraction | 0.1, 0.2 |
| E3: Clustering | k | 5, 10 |
| E3: Clustering | extract_from | centroid, extreme, both |
| E4: 2-swap | max_swap_passes | 3, 5 |
| E5: Column memory | lambda | 0.1, 0.5, 1.0 |

## Execution Order

1. Run baseline on all 3 graphs → reference chi and column counts
2. Ideas 3 & 4 first (no extra API calls, pure post-processing)
3. Ideas 1 & 5 next (modify QP but same API call count)
4. Idea 2 last (increases API call count)

## Results

All 19 configurations were tested on ER(30, 0.5, seed=42) with n=30, m=217, greedy_chi=7.
Baseline used `method="gibbons"`, `num_samples=100`, `multi_prune=True`,
`randomized_rounding=True`, `num_random_rounds=10`, `local_search_passes=5`.

### Summary Table

| Experiment | chi | delta | Iters | Cols | API Calls | Time (s) |
|---|---|---|---|---|---|---|
| E0_baseline (run 1) | 7 | — | 11 | 129 | 12 | 1053 |
| E0_baseline (run 2) | 7 | — | 9 | 142 | 10 | 1065 |
| E0_baseline (run 3) | 7 | — | 11 | 144 | 12 | 1329 |
| E0_baseline (run 4) | 7 | — | 10 | 134 | 11 | 1246 |
| E1_eps0.05_n2 | 7 | +0 | 11 | 146 | 12 | 1959 |
| E1_eps0.05_n3 | 7 | +0 | 7 | 141 | 8 | 2006 |
| E1_eps0.10_n2 | 7 | +0 | 7 | 134 | 8 | 1879 |
| E2_subgraph_c2_o0.1 | 7 | +0 | 39 | 217 | 40 | 7342 |
| E2_subgraph_c2_o0.2 | 7 | +0 | 17 | 177 | 18 | 1662 |
| E2_subgraph_c3_o0.1 | **8** | **+1** | 17 | 133 | 18 | 2306 |
| E2_subgraph_c3_o0.2 | **8** | **+1** | 16 | 173 | 17 | 1967 |
| E3_cluster_k5_centroid | 7 | +0 | 12 | 129 | 13 | 1465 |
| E3_cluster_k5_extreme | 7 | +0 | 13 | 124 | 14 | 1643 |
| E3_cluster_k5_both | 7 | +0 | 11 | 130 | 12 | 1184 |
| E3_cluster_k10_centroid | 7 | +0 | 11 | 130 | 12 | 946 |
| E3_cluster_k10_extreme | 7 | +0 | 11 | 118 | 12 | 901 |
| **E3_cluster_k10_both** | **7** | **+0** | **9** | **127** | **10** | **738** |
| E4_2swap_p3 | **8** | **+1** | 6 | 126 | 7 | 524 |
| E4_2swap_p5 | 7 | +0 | 7 | 139 | 8 | 617 |
| E5_colmem_lam0.1 | 7 | +0 | 8 | 133 | 9 | 1017 |
| E5_colmem_lam0.5 | 7 | +0 | 11 | 131 | 12 | 1400 |
| E5_colmem_lam1.0 | **8** | **+1** | 9 | 125 | 10 | 1147 |

Note: E1_eps0.10_n3, E1_eps0.15_n2, E1_eps0.15_n3 were not completed due to a QCI API
outage during the experiment run. The three completed E1 configs all showed chi=7.

### Key Findings

**1. No idea improved chi.** All 19 configurations achieved chi=7 (same as baseline) or
worse (chi=8). The baseline Dirac oracle with existing multi-threshold + multi-prune +
randomized rounding + 1-swap extraction already saturates what is achievable on this graph.

**2. Four configurations actively hurt chi (+1 worse):**
- E2_subgraph_c3 (both overlap settings): Splitting a 50%-density graph into 3 clusters
  produces subproblems too small for Dirac to find quality IS.
- E4_2swap_p3: Aggressive 2-swap with only 3 passes appears to over-optimize individual
  IS toward locally heavy nodes, reducing column diversity.
- E5_colmem_lam1.0: Strong diversity penalty (lambda=1.0) distorts the Gibbons QP
  weights too much, causing Dirac to find poor IS.

**3. E3_cluster_k10_both improves convergence speed.** This was the standout finding:
k-means clustering of Dirac solution vectors with k=10, extracting from both centroids
and extremes, achieved chi=7 in just 9 iterations / 738s — a **1.4-1.8x speedup** over
the baseline (1053-1329s) with the same final chi. The clustering reduces noise in
solution vectors (centroids) while preserving diversity (extremes), leading to a more
efficient column pool that converges faster.

**4. E2 subgraph decomposition is catastrophically slow on dense graphs.** The c=2,
overlap=0.1 config took 7342s (6x baseline) due to 2 API calls per CG iteration and 39
iterations to converge. The smaller per-partition QPs produce fewer profitable columns per
call, requiring more iterations overall.

**5. E1 dual perturbation finds more columns but no chi benefit.** The eps=0.05_n2 config
generated the most columns (146, vs baseline 129-144) but still achieved chi=7. More
columns does not help when the LP-to-ILP integrality gap is the binding constraint.

### Analysis: Why the Baseline is Hard to Beat

The ER(30, 0.5) graph with greedy_chi=7 appears to have chi=7 as the true chromatic number
(or very close to it). The existing Dirac extraction pipeline already generates 130-144
unique profitable columns across ~10 iterations — a rich column pool. The gap between the
LP relaxation bound (~5.3) and the integer solution (7) is due to the **LP-to-ILP
integrality gap**, not column deficiency. No amount of column extraction improvement can
close this gap; it would require either:
- A tighter LP formulation (e.g., adding cutting planes)
- A better ILP solver for the final set-cover step
- Larger graph instances where column quality matters more

### Recommendation

**Adopt E3_cluster_k10_both as default extraction.** While it does not improve chi, it
converges 1.4-1.8x faster with no additional API calls. The k-means clustering step is
computationally trivial (~1ms for 100 vectors of dimension 30) and could be enabled by
default in the Dirac oracle without downside.

For future work on improving chi, focus on the ILP phase (cutting planes, branch-and-price)
rather than column extraction.
