#!/usr/bin/env python
"""Benchmark: Dirac-3 vs classical column generation on a range of graphs.

Outputs a markdown table comparing chromatic number, iterations, columns
generated, and wall-clock time for both oracles.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.pricing.classical import ClassicalPricingOracle
from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle
from quantum_colgen.graphs import TEST_GRAPHS, KNOWN_CHROMATIC, erdos_renyi


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    oracle: str
    graph_name: str
    n_nodes: int
    n_edges: int
    expected_chi: Optional[int]
    chi: Optional[int]
    valid: bool
    iterations: int
    columns: int
    wall_seconds: float
    optimal: bool = False  # chi == expected_chi
    converged: bool = False
    max_iterations: int = 500


@dataclass
class BenchmarkSuite:
    results: List[RunResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_once(
    graph: nx.Graph,
    graph_name: str,
    oracle_name: str,
    expected_chi: Optional[int],
    dirac_kwargs: dict,
    verbose: bool = False,
    max_iterations: int = 500,
) -> RunResult:
    if oracle_name == "classical":
        oracle = ClassicalPricingOracle()
    else:
        oracle = DiracPricingOracle(**dirac_kwargs)

    t0 = time.time()
    num_colors, coloring, stats = column_generation(
        graph, oracle, max_iterations=max_iterations, verbose=verbose,
    )
    elapsed = time.time() - t0

    valid = verify_coloring(graph, coloring) if coloring else False
    opt = (num_colors == expected_chi) if expected_chi is not None and num_colors is not None else False

    iters = stats.get("iterations", 0)
    converged = iters < max_iterations

    return RunResult(
        oracle=oracle_name,
        graph_name=graph_name,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        expected_chi=expected_chi,
        chi=num_colors,
        valid=valid,
        iterations=iters,
        columns=stats.get("columns_generated", 0),
        wall_seconds=round(elapsed, 2),
        optimal=opt,
        converged=converged,
        max_iterations=max_iterations,
    )


def _nx_greedy_chi(G: nx.Graph) -> int:
    """NetworkX greedy upper bound on chromatic number."""
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    return max(coloring.values()) + 1 if coloring else 0


# ---------------------------------------------------------------------------
# Benchmark graph catalogue
# ---------------------------------------------------------------------------

def build_graph_catalogue(
    er_sizes: List[int],
    er_probs: List[float],
    seed: int = 42,
) -> List[Tuple[str, nx.Graph, Optional[int]]]:
    """Return (name, graph, expected_chi_or_None) tuples."""
    catalogue = []

    # 1. Predefined test graphs with known chromatic numbers
    for name, factory in TEST_GRAPHS.items():
        catalogue.append((name, factory(), KNOWN_CHROMATIC[name]))

    # 2. Erdos-Renyi graphs (chi unknown — use None)
    for n in er_sizes:
        for p in er_probs:
            G = erdos_renyi(n, p, seed=seed)
            if G.number_of_edges() == 0:
                continue
            label = f"ER({n},{p})"
            catalogue.append((label, G, None))

    return catalogue


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_markdown_table(results: List[RunResult]):
    hdr = (
        "| Graph | n | m | Oracle | chi | expected | optimal | valid "
        "| iters | cols | time(s) |"
    )
    sep = (
        "|-------|---|---|--------|-----|----------|---------|-------"
        "|-------|------|---------|"
    )
    print(hdr)
    print(sep)
    for r in results:
        exp = str(r.expected_chi) if r.expected_chi is not None else "-"
        opt = "Y" if r.optimal else ("?" if r.expected_chi is None else "N")
        print(
            f"| {r.graph_name:<16} | {r.n_nodes:>3} | {r.n_edges:>3} "
            f"| {r.oracle:<10} | {r.chi if r.chi else 'FAIL':>3} | {exp:>8} "
            f"| {opt:>7} | {str(r.valid):>5} "
            f"| {r.iterations:>5} | {r.columns:>4} | {r.wall_seconds:>7} |"
        )


def print_comparison_summary(results: List[RunResult]):
    """Side-by-side classical vs dirac per graph."""
    by_graph: Dict[str, Dict[str, RunResult]] = {}
    for r in results:
        by_graph.setdefault(r.graph_name, {})[r.oracle] = r

    print("\n## Comparison: Classical vs Dirac")
    print(
        "| Graph | n | m | chi(classical) | chi(dirac) | match "
        "| t_class(s) | t_dirac(s) | speedup |"
    )
    print(
        "|-------|---|---|----------------|------------|-------"
        "|------------|------------|---------|"
    )
    for gname, oracles in by_graph.items():
        c = oracles.get("classical")
        d = oracles.get("dirac")
        if not c or not d:
            continue
        match = "Y" if c.chi == d.chi else "N"
        speedup = f"{c.wall_seconds / d.wall_seconds:.2f}x" if d.wall_seconds > 0 else "-"
        print(
            f"| {gname:<16} | {c.n_nodes:>3} | {c.n_edges:>3} "
            f"| {c.chi if c.chi else 'FAIL':>14} | {d.chi if d.chi else 'FAIL':>10} | {match:>5} "
            f"| {c.wall_seconds:>10} | {d.wall_seconds:>10} | {speedup:>7} |"
        )


def print_greedy_comparison(catalogue, results: List[RunResult]):
    """Compare CG results against networkx greedy coloring."""
    print("\n## CG vs NetworkX Greedy Upper Bound")
    print(
        "| Graph | n | m | greedy | chi(classical) | chi(dirac) "
        "| CG_saves_classical | CG_saves_dirac |"
    )
    print(
        "|-------|---|---|--------|----------------|------------"
        "|--------------------|--------------------|"
    )
    by_graph: Dict[str, Dict[str, RunResult]] = {}
    for r in results:
        by_graph.setdefault(r.graph_name, {})[r.oracle] = r

    for gname, G, _ in catalogue:
        greedy = _nx_greedy_chi(G)
        oracles = by_graph.get(gname, {})
        c = oracles.get("classical")
        d = oracles.get("dirac")
        chi_c = c.chi if c and c.chi else "-"
        chi_d = d.chi if d and d.chi else "-"
        save_c = greedy - c.chi if c and c.chi else "-"
        save_d = greedy - d.chi if d and d.chi else "-"
        print(
            f"| {gname:<16} | {G.number_of_nodes():>3} | {G.number_of_edges():>3} "
            f"| {greedy:>6} | {chi_c:>14} | {chi_d:>10} "
            f"| {save_c:>18} | {save_d:>18} |"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark Dirac vs Classical CG")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--er-sizes",
        type=int,
        nargs="+",
        default=[8, 10, 12, 15],
        help="Erdos-Renyi node counts (default: 8 10 12 15)",
    )
    parser.add_argument(
        "--er-probs",
        type=float,
        nargs="+",
        default=[0.3, 0.5],
        help="Erdos-Renyi edge probabilities (default: 0.3 0.5)",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--relax-schedule", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--method", default="gibbons", choices=["gibbons", "filter"])
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--max-iterations", type=int, default=500, help="Max CG iterations (default: 500)")
    parser.add_argument("--classical-only", action="store_true", help="Skip Dirac runs")
    parser.add_argument("--dirac-only", action="store_true", help="Skip classical runs")

    args = parser.parse_args()

    dirac_kwargs = {
        "method": args.method,
        "num_samples": args.num_samples,
        "relax_schedule": args.relax_schedule,
    }

    catalogue = build_graph_catalogue(args.er_sizes, args.er_probs)
    all_results: List[RunResult] = []

    total = len(catalogue)
    oracles = []
    if not args.dirac_only:
        oracles.append("classical")
    if not args.classical_only:
        oracles.append("dirac")

    for idx, (gname, G, expected) in enumerate(catalogue, 1):
        for oracle_name in oracles:
            tag = f"[{idx}/{total}] {gname} ({oracle_name})"
            print(f"\n{'='*60}")
            print(f"  {tag}  n={G.number_of_nodes()} m={G.number_of_edges()}")
            print(f"{'='*60}")
            result = _run_once(
                G, gname, oracle_name, expected, dirac_kwargs,
                verbose=args.verbose, max_iterations=args.max_iterations,
            )
            all_results.append(result)
            status = "PASS" if result.valid else "FAIL"
            chi_str = str(result.chi) if result.chi else "FAIL"
            conv = "converged" if result.converged else f"HIT_LIMIT({result.iterations})"
            print(f"  => chi={chi_str}  valid={result.valid}  time={result.wall_seconds}s  iters={result.iterations}  {conv}  [{status}]")

    # Print summary tables
    print("\n\n" + "=" * 80)
    print("# Benchmark Results")
    print("=" * 80)
    print("\n## All Runs")
    print_markdown_table(all_results)

    if "classical" in oracles and "dirac" in oracles:
        print_comparison_summary(all_results)

    print_greedy_comparison(catalogue, all_results)

    # Save JSON if requested
    if args.json:
        with open(args.json, "w") as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
        print(f"\nResults saved to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
