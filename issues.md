# Known Issues and Scalability Limitations

## Dirac CG Scalability at n=200

Benchmark run on ER(200, 0.3) graph (200 nodes, 5918 edges) revealed significant scalability issues.

### Benchmark Results

| Method | Colors (chi) | Wall Time | Status |
|--------|--------------|-----------|--------|
| Greedy (networkx `largest_first`) | 22 | 0.0004s | Completed |
| Dirac CG | - | >10.75 hours | **Did not complete** |

### Dirac CG Breakdown

| Phase | Metric | Value |
|-------|--------|-------|
| CG iterations | API calls | 190 |
| CG phase | Wall time | ~5.5 hours |
| CG phase | Avg time/API call | ~1.74 minutes |
| Final ILP | Wall time | >5 hours (stopped) |
| Final ILP | Status | **Did not complete** |

### Root Causes

1. **Dirac API latency**: Each API call takes ~1.5-2 minutes, making the CG phase slow for problems requiring many iterations.

2. **ILP solver bottleneck**: The final set-cover ILP becomes intractable with hundreds of columns.
   - Solver: SciPy's `milp()` using HiGHS
   - Problem size: ~190+ binary variables, 200 equality constraints
   - Set-cover ILP is NP-hard; branch-and-bound tree explodes

3. **Column count growth**: At n=200, CG generates many columns before convergence, exacerbating the ILP complexity.

### Potential Mitigations

1. **Use commercial ILP solver**: Gurobi or CPLEX are often 10-100x faster than HiGHS for MIP problems.

2. **Add ILP time limit**: Return best feasible solution found within a time budget.

3. **LP rounding heuristics**: Instead of exact ILP, use randomized rounding on the LP solution.

4. **Column management**: Limit the number of columns added per iteration or prune low-quality columns.

5. **Reduce Dirac API calls**: Batch multiple pricing problems or use warm-starting techniques.

### Practical Limits

Based on this benchmark and previous results:

| Graph Size | Dirac CG Feasibility |
|------------|---------------------|
| n <= 30 | Practical (~750-1450s per graph) |
| n = 50-100 | Marginal (hours per graph) |
| n >= 200 | **Not practical** with current implementation |

Greedy coloring remains orders of magnitude faster at all scales, though it may use more colors than optimal.
