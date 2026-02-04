#!/usr/bin/env python
"""Benchmark: Improved Dirac CG vs classical CG vs networkx greedy at scale.

Extends the original benchmark to support n=100-200 ER graphs and compares
column generation (classical and/or Dirac) against networkx greedy coloring.
Includes per-oracle timing breakdown (API call time, extraction time, etc.).
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
from quantum_colgen.graphs import erdos_renyi


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TimingBreakdown:
    """Per-oracle timing breakdown from OracleTimer."""
    num_api_calls: int = 0
    total_api_seconds: float = 0.0
    total_extract_seconds: float = 0.0
    avg_api_seconds: float = 0.0
    avg_columns_per_call: float = 0.0
    total_columns_found: int = 0


@dataclass
class RunResult:
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
    max_iterations: int = 500
    timing: Optional[TimingBreakdown] = None


@dataclass
class GreedyResult:
    graph_name: str
    n_nodes: int
    n_edges: int
    chi: int
    wall_seconds: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cg(
    graph: nx.Graph,
    graph_name: str,
    oracle_name: str,
    dirac_kwargs: dict,
    verbose: bool = False,
    max_iterations: int = 500,
) -> RunResult:
    if oracle_name == "classical":
        oracle = ClassicalPricingOracle()
    else:
        from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle
        oracle = DiracPricingOracle(**dirac_kwargs)

    t0 = time.time()
    num_colors, coloring, stats = column_generation(
        graph, oracle, max_iterations=max_iterations, verbose=verbose,
    )
    elapsed = time.time() - t0

    valid = verify_coloring(graph, coloring) if coloring else False
    iters = stats.get("iterations", 0)
    converged = iters < max_iterations

    # Extract timing from oracle
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
        max_iterations=max_iterations,
        timing=timing,
    )


def _run_greedy(graph: nx.Graph, graph_name: str) -> GreedyResult:
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


# ---------------------------------------------------------------------------
# Graph catalogue
# ---------------------------------------------------------------------------

def build_catalogue(
    er_sizes: List[int],
    er_probs: List[float],
    seed: int = 42,
) -> List[Tuple[str, nx.Graph]]:
    catalogue = []
    for n in er_sizes:
        for p in er_probs:
            G = erdos_renyi(n, p, seed=seed)
            if G.number_of_edges() == 0:
                continue
            label = f"ER({n},{p})"
            catalogue.append((label, G))
    return catalogue


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _index_results(
    cg_results: List[RunResult],
    greedy_results: List[GreedyResult],
) -> Tuple[Dict[str, GreedyResult], Dict[str, Dict[str, RunResult]], List[str]]:
    """Build lookup dicts and ordered name list."""
    greedy_by_name = {g.graph_name: g for g in greedy_results}
    cg_by_name: Dict[str, Dict[str, RunResult]] = {}
    for r in cg_results:
        cg_by_name.setdefault(r.graph_name, {})[r.oracle] = r
    all_names = []
    seen = set()
    for r in greedy_results:
        if r.graph_name not in seen:
            all_names.append(r.graph_name)
            seen.add(r.graph_name)
    return greedy_by_name, cg_by_name, all_names


def print_results_table(
    cg_results: List[RunResult],
    greedy_results: List[GreedyResult],
):
    greedy_by_name, cg_by_name, all_names = _index_results(cg_results, greedy_results)

    print("\n## Benchmark Results")
    hdr = "| Graph | n | m | Greedy | Classical CG | Dirac CG | Classical time(s) | Dirac time(s) | Classical iters | Dirac iters |"
    sep = "|-------|---|---|--------|-------------|----------|-------------------|---------------|-----------------|-------------|"
    print(hdr)
    print(sep)

    for gname in all_names:
        g = greedy_by_name.get(gname)
        oracles = cg_by_name.get(gname, {})
        c = oracles.get("classical")
        d = oracles.get("dirac")

        n = g.n_nodes if g else "-"
        m = g.n_edges if g else "-"
        greedy_chi = str(g.chi) if g else "-"
        c_chi = str(c.chi) if c and c.chi else "-"
        d_chi = str(d.chi) if d and d.chi else "-"
        c_time = f"{c.wall_seconds}" if c else "-"
        d_time = f"{d.wall_seconds}" if d else "-"
        c_iters = str(c.iterations) if c else "-"
        d_iters = str(d.iterations) if d else "-"

        print(
            f"| {gname:<16} | {n:>3} | {m:>5} "
            f"| {greedy_chi:>6} | {c_chi:>11} | {d_chi:>8} "
            f"| {c_time:>17} | {d_time:>13} "
            f"| {c_iters:>15} | {d_iters:>11} |"
        )


def print_timing_table(cg_results: List[RunResult]):
    """Print per-oracle timing breakdown."""
    print("\n## Oracle Timing Breakdown")
    hdr = (
        "| Graph | Oracle | API calls | Total API(s) | Avg API(s) "
        "| Extract(s) | Cols found | Avg cols/call | Wall(s) |"
    )
    sep = (
        "|-------|--------|-----------|-------------|----------"
        "|------------|------------|---------------|---------|"
    )
    print(hdr)
    print(sep)

    for r in cg_results:
        t = r.timing
        if t is None:
            continue
        print(
            f"| {r.graph_name:<16} | {r.oracle:<10} "
            f"| {t.num_api_calls:>9} | {t.total_api_seconds:>11.2f} | {t.avg_api_seconds:>8.2f} "
            f"| {t.total_extract_seconds:>10.4f} | {t.total_columns_found:>10} "
            f"| {t.avg_columns_per_call:>13.2f} | {r.wall_seconds:>7.2f} |"
        )


def print_savings_table(
    cg_results: List[RunResult],
    greedy_results: List[GreedyResult],
):
    greedy_by_name, cg_by_name, _ = _index_results(cg_results, greedy_results)

    print("\n## Colors Saved vs Greedy")
    print("| Graph | n | Greedy | Classical CG | Saved(classical) | Dirac CG | Saved(dirac) |")
    print("|-------|---|--------|-------------|------------------|----------|--------------|")

    for g in greedy_results:
        oracles = cg_by_name.get(g.graph_name, {})
        c = oracles.get("classical")
        d = oracles.get("dirac")

        save_c = str(g.chi - c.chi) if c and c.chi else "-"
        save_d = str(g.chi - d.chi) if d and d.chi else "-"
        c_chi = str(c.chi) if c and c.chi else "-"
        d_chi = str(d.chi) if d and d.chi else "-"

        print(
            f"| {g.graph_name:<16} | {g.n_nodes:>3} "
            f"| {g.chi:>6} | {c_chi:>11} | {save_c:>16} "
            f"| {d_chi:>8} | {save_d:>12} |"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark improved Dirac CG vs classical CG vs greedy"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--er-sizes", type=int, nargs="+", default=[100, 125, 150, 175, 200],
        help="Erdos-Renyi node counts (default: 100 125 150 175 200)",
    )
    parser.add_argument(
        "--er-probs", type=float, nargs="+", default=[0.3],
        help="Erdos-Renyi edge probabilities (default: 0.3)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--json", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--classical-only", action="store_true", help="Skip Dirac runs")
    parser.add_argument("--dirac-only", action="store_true", help="Skip classical CG runs")
    parser.add_argument("--dirac", action="store_true", help="Include Dirac runs")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--relax-schedule", type=int, default=2, choices=[1, 2, 3, 4])
    parser.add_argument("--method", default="gibbons", choices=["gibbons", "filter"])

    args = parser.parse_args()

    dirac_kwargs = {
        "method": args.method,
        "num_samples": args.num_samples,
        "relax_schedule": args.relax_schedule,
    }

    catalogue = build_catalogue(args.er_sizes, args.er_probs, seed=args.seed)

    if args.dirac_only:
        oracles = ["dirac"]
    elif args.classical_only:
        oracles = ["classical"]
    elif args.dirac:
        oracles = ["classical", "dirac"]
    else:
        oracles = ["classical"]

    all_cg: List[RunResult] = []
    all_greedy: List[GreedyResult] = []

    total = len(catalogue)
    for idx, (gname, G) in enumerate(catalogue, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{total}] {gname}  n={G.number_of_nodes()} m={G.number_of_edges()}")
        print(f"{'='*60}")

        # Greedy baseline
        gr = _run_greedy(G, gname)
        all_greedy.append(gr)
        print(f"  Greedy: chi={gr.chi}  time={gr.wall_seconds}s")

        # CG runs
        for oracle_name in oracles:
            print(f"  Running {oracle_name} CG ...", flush=True)
            result = _run_cg(
                G, gname, oracle_name, dirac_kwargs,
                verbose=args.verbose, max_iterations=args.max_iterations,
            )
            all_cg.append(result)
            conv = "converged" if result.converged else f"HIT_LIMIT({result.iterations})"
            status = "PASS" if result.valid else "FAIL"
            t = result.timing
            timing_str = ""
            if t and t.num_api_calls > 0:
                timing_str = (
                    f"  api_calls={t.num_api_calls}  "
                    f"api_time={t.total_api_seconds:.2f}s  "
                    f"avg_api={t.avg_api_seconds:.2f}s/call  "
                    f"cols_found={t.total_columns_found}  "
                    f"avg_cols/call={t.avg_columns_per_call:.1f}"
                )
            print(
                f"  {oracle_name}: chi={result.chi}  time={result.wall_seconds}s  "
                f"iters={result.iterations}  cols={result.columns}  {conv}  [{status}]"
            )
            if timing_str:
                print(timing_str)

    # Summary tables
    print("\n\n" + "=" * 80)
    print_results_table(all_cg, all_greedy)
    print_timing_table(all_cg)
    print_savings_table(all_cg, all_greedy)

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
