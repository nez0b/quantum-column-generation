#!/usr/bin/env python
"""Benchmark: HiGHS vs Hexaly for final set-cover ILP.

This script compares the performance of HiGHS and Hexaly solvers on the
final set-cover ILP phase of column generation. It:
1. Generates columns using classical CG
2. Times the final ILP phase with both solvers
3. Compares solution quality and runtime
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import networkx as nx

from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.master_problem import solve_final_ilp, HEXALY_AVAILABLE
from quantum_colgen.pricing.classical import ClassicalPricingOracle
from quantum_colgen.graphs import erdos_renyi


@dataclass
class ILPBenchmarkResult:
    """Result from benchmarking a single ILP solver run."""
    graph_name: str
    n_nodes: int
    n_edges: int
    num_columns: int
    solver: str
    chi: Optional[int]
    valid: bool
    ilp_seconds: float
    timed_out: bool = False
    time_limit: Optional[int] = None


@dataclass
class CGPrepResult:
    """Result from column generation phase (shared between solvers)."""
    graph_name: str
    n_nodes: int
    n_edges: int
    num_columns: int
    cg_seconds: float
    cg_iterations: int


def _run_cg_only(
    graph: nx.Graph,
    graph_name: str,
    max_iterations: int = 500,
    verbose: bool = False,
) -> Tuple[List, int, float, int]:
    """Run CG to generate columns without solving final ILP.

    Returns: (columns, num_vertices, elapsed_seconds, iterations)
    """
    oracle = ClassicalPricingOracle()

    # We need to duplicate the CG loop logic to capture columns before ILP
    from quantum_colgen.master_problem import solve_rmp

    node_list = sorted(graph.nodes())
    num_vertices = len(node_list)

    # Initialize with singleton columns
    columns = [frozenset([i]) for i in range(num_vertices)]
    known_sigs = {tuple(sorted(c)) for c in columns}

    t0 = time.time()
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        iterations = iteration

        # Solve RMP
        obj, dual_vars = solve_rmp(columns, num_vertices)
        if dual_vars is None:
            break

        # Pricing subproblem
        new_cols = oracle.solve(graph, dual_vars)

        if not new_cols:
            break

        added = 0
        for col_set in new_cols:
            sig = tuple(sorted(col_set))
            if sig not in known_sigs:
                columns.append(frozenset(col_set))
                known_sigs.add(sig)
                added += 1

        if added == 0:
            break

        if verbose:
            print(f"    Iter {iteration}: {len(columns)} columns")

    elapsed = time.time() - t0
    return columns, num_vertices, elapsed, iterations


def _benchmark_ilp_solver(
    columns: List,
    num_vertices: int,
    graph: nx.Graph,
    graph_name: str,
    solver: str,
    time_limit: Optional[int] = None,
) -> ILPBenchmarkResult:
    """Benchmark a single ILP solver."""
    t0 = time.time()
    num_colors, selected = solve_final_ilp(
        columns, num_vertices, solver=solver, time_limit=time_limit
    )
    elapsed = time.time() - t0

    # Verify coloring
    valid = False
    if num_colors is not None and selected:
        coloring = [columns[i] for i in selected]
        valid = verify_coloring(graph, coloring)

    # Detect timeout (heuristic: if we hit the limit and didn't find optimal)
    timed_out = time_limit is not None and elapsed >= time_limit * 0.95

    return ILPBenchmarkResult(
        graph_name=graph_name,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        num_columns=len(columns),
        solver=solver,
        chi=num_colors,
        valid=valid,
        ilp_seconds=round(elapsed, 3),
        timed_out=timed_out,
        time_limit=time_limit,
    )


def build_catalogue(
    er_sizes: List[int],
    er_probs: List[float],
    seed: int = 42,
) -> List[Tuple[str, nx.Graph]]:
    """Build graph catalogue."""
    catalogue = []
    for n in er_sizes:
        for p in er_probs:
            G = erdos_renyi(n, p, seed=seed)
            if G.number_of_edges() == 0:
                continue
            label = f"ER({n},{p})"
            catalogue.append((label, G))
    return catalogue


def print_results_table(
    cg_results: List[CGPrepResult],
    ilp_results: List[ILPBenchmarkResult],
):
    """Print comparison table."""
    # Index ILP results by (graph_name, solver)
    ilp_by_key = {(r.graph_name, r.solver): r for r in ilp_results}
    cg_by_name = {r.graph_name: r for r in cg_results}

    print("\n## ILP Solver Benchmark Results")
    print("| Graph | n | m | Columns | CG(s) | HiGHS chi | HiGHS(s) | Hexaly chi | Hexaly(s) | Match |")
    print("|-------|---|---|---------|-------|-----------|----------|------------|-----------|-------|")

    for cg in cg_results:
        highs = ilp_by_key.get((cg.graph_name, "highs"))
        hexaly = ilp_by_key.get((cg.graph_name, "hexaly"))

        highs_chi = str(highs.chi) if highs and highs.chi else "-"
        highs_time = f"{highs.ilp_seconds}" if highs else "-"
        if highs and highs.timed_out:
            highs_time += "*"

        hexaly_chi = str(hexaly.chi) if hexaly and hexaly.chi else "-"
        hexaly_time = f"{hexaly.ilp_seconds}" if hexaly else "-"
        if hexaly and hexaly.timed_out:
            hexaly_time += "*"

        # Check if results match
        match = "-"
        if highs and hexaly and highs.chi is not None and hexaly.chi is not None:
            match = "Yes" if highs.chi == hexaly.chi else "No"

        print(
            f"| {cg.graph_name:<16} | {cg.n_nodes:>3} | {cg.n_edges:>5} "
            f"| {cg.num_columns:>7} | {cg.cg_seconds:>5.1f} "
            f"| {highs_chi:>9} | {highs_time:>8} "
            f"| {hexaly_chi:>10} | {hexaly_time:>9} | {match:>5} |"
        )

    print("\n* = hit time limit")


def print_speedup_table(
    cg_results: List[CGPrepResult],
    ilp_results: List[ILPBenchmarkResult],
):
    """Print speedup comparison."""
    ilp_by_key = {(r.graph_name, r.solver): r for r in ilp_results}

    print("\n## Speedup Analysis")
    print("| Graph | Columns | HiGHS(s) | Hexaly(s) | Speedup |")
    print("|-------|---------|----------|-----------|---------|")

    for cg in cg_results:
        highs = ilp_by_key.get((cg.graph_name, "highs"))
        hexaly = ilp_by_key.get((cg.graph_name, "hexaly"))

        if not highs or not hexaly:
            continue

        speedup = "-"
        if highs.ilp_seconds > 0 and hexaly.ilp_seconds > 0:
            ratio = highs.ilp_seconds / hexaly.ilp_seconds
            if ratio > 1:
                speedup = f"{ratio:.1f}x (Hexaly)"
            else:
                speedup = f"{1/ratio:.1f}x (HiGHS)"

        print(
            f"| {cg.graph_name:<16} | {cg.num_columns:>7} "
            f"| {highs.ilp_seconds:>8.3f} | {hexaly.ilp_seconds:>9.3f} | {speedup:>7} |"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HiGHS vs Hexaly for final set-cover ILP"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--er-sizes", type=int, nargs="+", default=[30, 50, 75, 100],
        help="Erdos-Renyi node counts (default: 30 50 75 100)",
    )
    parser.add_argument(
        "--er-probs", type=float, nargs="+", default=[0.3],
        help="Erdos-Renyi edge probabilities (default: 0.3)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument(
        "--time-limit", type=int, default=None,
        help="ILP time limit in seconds per solver"
    )
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON")
    parser.add_argument(
        "--highs-only", action="store_true",
        help="Only run HiGHS (skip Hexaly)"
    )
    parser.add_argument(
        "--hexaly-only", action="store_true",
        help="Only run Hexaly (skip HiGHS)"
    )

    args = parser.parse_args()

    # Determine which solvers to run
    solvers = []
    if not args.hexaly_only:
        solvers.append("highs")
    if not args.highs_only:
        if HEXALY_AVAILABLE:
            solvers.append("hexaly")
        else:
            print("Warning: Hexaly not installed, skipping Hexaly benchmarks")
            print("Install with: uv pip install hexaly")

    if not solvers:
        print("Error: No solvers available to benchmark")
        return 1

    catalogue = build_catalogue(args.er_sizes, args.er_probs, seed=args.seed)

    all_cg: List[CGPrepResult] = []
    all_ilp: List[ILPBenchmarkResult] = []

    total = len(catalogue)
    for idx, (gname, G) in enumerate(catalogue, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{total}] {gname}  n={G.number_of_nodes()} m={G.number_of_edges()}")
        print(f"{'='*60}")

        # Phase 1: Column generation
        print("  Phase 1: Column generation ...", flush=True)
        columns, num_vertices, cg_elapsed, cg_iters = _run_cg_only(
            G, gname, max_iterations=args.max_iterations, verbose=args.verbose
        )

        cg_result = CGPrepResult(
            graph_name=gname,
            n_nodes=G.number_of_nodes(),
            n_edges=G.number_of_edges(),
            num_columns=len(columns),
            cg_seconds=round(cg_elapsed, 2),
            cg_iterations=cg_iters,
        )
        all_cg.append(cg_result)
        print(f"    Generated {len(columns)} columns in {cg_elapsed:.2f}s ({cg_iters} iterations)")

        # Phase 2: Benchmark ILP solvers
        for solver in solvers:
            print(f"  Phase 2: Solving ILP with {solver} ...", flush=True)
            result = _benchmark_ilp_solver(
                columns, num_vertices, G, gname, solver,
                time_limit=args.time_limit,
            )
            all_ilp.append(result)

            status = "PASS" if result.valid else "FAIL"
            timeout_str = " (timeout)" if result.timed_out else ""
            print(
                f"    {solver}: chi={result.chi}  time={result.ilp_seconds}s{timeout_str}  [{status}]"
            )

    # Summary tables
    print("\n\n" + "=" * 80)
    print_results_table(all_cg, all_ilp)

    if "highs" in solvers and "hexaly" in solvers:
        print_speedup_table(all_cg, all_ilp)

    # Save JSON
    if args.json:
        output = {
            "cg_results": [asdict(r) for r in all_cg],
            "ilp_results": [asdict(r) for r in all_ilp],
            "config": {
                "er_sizes": args.er_sizes,
                "er_probs": args.er_probs,
                "seed": args.seed,
                "max_iterations": args.max_iterations,
                "time_limit": args.time_limit,
            }
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
