#!/usr/bin/env python
"""Benchmark: Classical MILP vs Classical LP vs Dirac oracles.

Compares three pricing oracle approaches:
1. Classical MILP (exact, NP-hard) - current default
2. Classical LP (relaxed, polynomial) + extraction - new
3. Dirac QP (Motzkin-Straus) + extraction - quantum

All use column generation for graph coloring with final ILP.

Usage:
    # Classical only (no Dirac API calls)
    uv run python scripts/benchmark_oracles.py --classical-only --er-sizes 20 30 40

    # Full comparison including Dirac
    source ~/.zshrc
    uv run python scripts/benchmark_oracles.py --dirac --er-sizes 20 30

    # Save results
    uv run python scripts/benchmark_oracles.py --classical-only --er-sizes 30 50 --json results/oracle_benchmark.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import networkx as nx

from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.pricing.classical import ClassicalPricingOracle
from quantum_colgen.pricing.classical_lp import ClassicalLPPricingOracle
from quantum_colgen.graphs import erdos_renyi


# Try to import Dirac
try:
    from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle, DIRAC_AVAILABLE
except ImportError:
    DIRAC_AVAILABLE = False


@dataclass
class TimingBreakdown:
    """Per-oracle timing breakdown."""
    num_api_calls: int = 0
    total_api_seconds: float = 0.0
    total_extract_seconds: float = 0.0
    avg_api_seconds: float = 0.0
    avg_columns_per_call: float = 0.0
    total_columns_found: int = 0


@dataclass
class RunResult:
    """Result from a single CG run."""
    oracle: str
    graph_name: str
    n_nodes: int
    n_edges: int
    chi: Optional[int]
    valid: bool
    iterations: int
    columns: int
    wall_seconds: float
    converged: bool = False
    timing: Optional[TimingBreakdown] = None


@dataclass
class GreedyResult:
    """Result from greedy coloring."""
    graph_name: str
    n_nodes: int
    n_edges: int
    chi: int
    wall_seconds: float


def run_greedy(graph: nx.Graph, graph_name: str) -> GreedyResult:
    """Run greedy coloring."""
    t0 = time.time()
    coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
    elapsed = time.time() - t0
    chi = max(coloring.values()) + 1 if coloring else 0
    return GreedyResult(
        graph_name=graph_name,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        chi=chi,
        wall_seconds=round(elapsed, 4),
    )


def run_cg(
    graph: nx.Graph,
    graph_name: str,
    oracle_name: str,
    max_iterations: int = 500,
    verbose: bool = False,
    dirac_enhanced: bool = True,
) -> RunResult:
    """Run column generation with specified oracle."""
    if oracle_name == "milp":
        oracle = ClassicalPricingOracle()
    elif oracle_name == "lp":
        oracle = ClassicalLPPricingOracle(
            multi_prune=True,
            randomized_rounding=True,
            num_random_rounds=10,
        )
    elif oracle_name == "dirac":
        if not DIRAC_AVAILABLE:
            raise RuntimeError("Dirac not available")
        if dirac_enhanced:
            oracle = DiracPricingOracle(
                method="gibbons",
                num_samples=100,
                multi_prune=True,
                randomized_rounding=True,
                num_random_rounds=10,
            )
        else:
            oracle = DiracPricingOracle(method="gibbons", num_samples=100)
    else:
        raise ValueError(f"Unknown oracle: {oracle_name}")

    t0 = time.time()
    num_colors, coloring, stats = column_generation(
        graph, oracle, max_iterations=max_iterations, verbose=verbose,
    )
    elapsed = time.time() - t0

    valid = verify_coloring(graph, coloring) if coloring else False
    iters = stats.get("iterations", 0)
    converged = iters < max_iterations

    # Extract timing
    timing_summary = oracle.timer.summary()
    timing = TimingBreakdown(**timing_summary)

    return RunResult(
        oracle=oracle_name,
        graph_name=graph_name,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        chi=num_colors,
        valid=valid,
        iterations=iters,
        columns=stats.get("columns_generated", 0),
        wall_seconds=round(elapsed, 2),
        converged=converged,
        timing=timing,
    )


def build_catalogue(
    er_sizes: List[int],
    er_probs: List[float],
    seed: int = 42,
) -> List[tuple]:
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
    cg_results: List[RunResult],
    greedy_results: List[GreedyResult],
):
    """Print comparison table."""
    greedy_by_name = {g.graph_name: g for g in greedy_results}
    cg_by_name: Dict[str, Dict[str, RunResult]] = {}
    for r in cg_results:
        cg_by_name.setdefault(r.graph_name, {})[r.oracle] = r

    print("\n## Benchmark Results: chi comparison")
    print("| Graph | n | m | Greedy | MILP | LP | Dirac | MILP time | LP time | Dirac time |")
    print("|-------|---|---|--------|------|-----|-------|-----------|---------|------------|")

    for g in greedy_results:
        oracles = cg_by_name.get(g.graph_name, {})
        milp = oracles.get("milp")
        lp = oracles.get("lp")
        dirac = oracles.get("dirac")

        milp_chi = str(milp.chi) if milp and milp.chi else "-"
        lp_chi = str(lp.chi) if lp and lp.chi else "-"
        dirac_chi = str(dirac.chi) if dirac and dirac.chi else "-"
        milp_time = f"{milp.wall_seconds}" if milp else "-"
        lp_time = f"{lp.wall_seconds}" if lp else "-"
        dirac_time = f"{dirac.wall_seconds}" if dirac else "-"

        print(
            f"| {g.graph_name:<12} | {g.n_nodes:>3} | {g.n_edges:>5} "
            f"| {g.chi:>6} | {milp_chi:>4} | {lp_chi:>3} | {dirac_chi:>5} "
            f"| {milp_time:>9} | {lp_time:>7} | {dirac_time:>10} |"
        )


def print_extraction_table(cg_results: List[RunResult]):
    """Print extraction efficiency comparison."""
    print("\n## Extraction Efficiency: columns per API call")
    print("| Graph | Oracle | API calls | Cols found | Cols/call | Iterations | Total cols |")
    print("|-------|--------|-----------|------------|-----------|------------|------------|")

    for r in cg_results:
        t = r.timing
        if t is None:
            continue
        print(
            f"| {r.graph_name:<12} | {r.oracle:<6} "
            f"| {t.num_api_calls:>9} | {t.total_columns_found:>10} "
            f"| {t.avg_columns_per_call:>9.1f} | {r.iterations:>10} | {r.columns:>10} |"
        )


def print_summary(
    cg_results: List[RunResult],
    greedy_results: List[GreedyResult],
):
    """Print summary analysis."""
    print("\n## Summary Analysis")

    # Group by oracle
    by_oracle: Dict[str, List[RunResult]] = {}
    for r in cg_results:
        by_oracle.setdefault(r.oracle, []).append(r)

    greedy_by_name = {g.graph_name: g for g in greedy_results}

    for oracle_name, results in by_oracle.items():
        wins = 0
        ties = 0
        losses = 0
        for r in results:
            greedy_chi = greedy_by_name[r.graph_name].chi
            if r.chi is None:
                continue
            if r.chi < greedy_chi:
                wins += 1
            elif r.chi == greedy_chi:
                ties += 1
            else:
                losses += 1

        total = wins + ties + losses
        if total > 0:
            print(f"\n{oracle_name.upper()} vs Greedy: {wins} wins, {ties} ties, {losses} losses")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MILP vs LP vs Dirac pricing oracles"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--er-sizes", type=int, nargs="+", default=[20, 30, 40],
        help="Erdos-Renyi node counts",
    )
    parser.add_argument(
        "--er-probs", type=float, nargs="+", default=[0.3, 0.5],
        help="Erdos-Renyi edge probabilities",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--json", type=str, help="Save results to JSON")
    parser.add_argument("--classical-only", action="store_true", help="Skip Dirac")
    parser.add_argument("--dirac", action="store_true", help="Include Dirac")
    parser.add_argument("--dirac-only", action="store_true", help="Only run Dirac")

    args = parser.parse_args()

    if args.dirac_only:
        oracles = ["dirac"]
    elif args.classical_only:
        oracles = ["milp", "lp"]
    elif args.dirac:
        oracles = ["milp", "lp", "dirac"]
    else:
        oracles = ["milp", "lp"]

    if "dirac" in oracles and not DIRAC_AVAILABLE:
        print("WARNING: Dirac not available, skipping")
        oracles = [o for o in oracles if o != "dirac"]

    catalogue = build_catalogue(args.er_sizes, args.er_probs, seed=args.seed)

    all_cg: List[RunResult] = []
    all_greedy: List[GreedyResult] = []

    total = len(catalogue)
    for idx, (gname, G) in enumerate(catalogue, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{total}] {gname}  n={G.number_of_nodes()} m={G.number_of_edges()}")
        print(f"{'='*60}")

        # Greedy baseline
        gr = run_greedy(G, gname)
        all_greedy.append(gr)
        print(f"  Greedy: chi={gr.chi}")

        # CG runs
        for oracle_name in oracles:
            print(f"  Running {oracle_name} CG ...", flush=True)
            result = run_cg(
                G, gname, oracle_name,
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
            all_cg.append(result)
            conv = "converged" if result.converged else f"HIT_LIMIT({result.iterations})"
            status = "PASS" if result.valid else "FAIL"
            t = result.timing
            cols_info = ""
            if t and t.num_api_calls > 0:
                cols_info = f"  ({t.avg_columns_per_call:.1f} cols/call)"
            print(
                f"    {oracle_name}: chi={result.chi}  iters={result.iterations}  "
                f"cols={result.columns}  time={result.wall_seconds}s  {conv}  [{status}]{cols_info}"
            )

    # Summary tables
    print("\n\n" + "=" * 80)
    print_results_table(all_cg, all_greedy)
    print_extraction_table(all_cg)
    print_summary(all_cg, all_greedy)

    # Save JSON
    if args.json:
        output = {
            "cg_results": [asdict(r) for r in all_cg],
            "greedy_results": [asdict(r) for r in all_greedy],
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
