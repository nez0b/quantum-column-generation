#!/usr/bin/env python3
"""Benchmark exact ILP solvers for minimum vertex coloring.

Compares HiGHS (scipy) and Hexaly on benchmark graphs.

Usage:
    uv run python scripts/benchmark_exact_ilp.py
    uv run python scripts/benchmark_exact_ilp.py --graphs er40_0.3 er50_0.3
    uv run python scripts/benchmark_exact_ilp.py --time-limit 300 --hexaly
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import networkx as nx

from quantum_colgen.direct_ilp import (
    solve_coloring_ilp_highs,
    solve_coloring_ilp_hexaly,
    validate_coloring,
    greedy_chromatic_upper_bound,
)


BENCHMARK_DIR = Path(__file__).parent.parent / "benchmarks"


def load_benchmark_graph(graph_name: str) -> nx.Graph:
    """Load a graph from benchmarks directory."""
    graph_path = BENCHMARK_DIR / graph_name / "graph.json"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    with open(graph_path) as f:
        data = json.load(f)

    # Custom format: {"n_nodes": N, "edges": [[u, v], ...]}
    n = data["n_nodes"]
    edges = data["edges"]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    return G


def list_benchmark_graphs() -> list:
    """List available benchmark graphs."""
    graphs = []
    if BENCHMARK_DIR.exists():
        for d in BENCHMARK_DIR.iterdir():
            if d.is_dir() and (d / "graph.json").exists():
                graphs.append(d.name)
    return sorted(graphs)


def save_result(graph_name: str, solver: str, result: dict):
    """Save benchmark result to JSON."""
    output_dir = BENCHMARK_DIR / graph_name
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"exact_ilp_{solver.lower()}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Saved: {output_path}")


def run_benchmark(
    graph_name: str,
    graph: nx.Graph,
    time_limit: float,
    run_hexaly: bool,
) -> dict:
    """Run benchmark on a single graph."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    greedy_chi = greedy_chromatic_upper_bound(graph)

    print(f"\n{'='*60}")
    print(f"  {graph_name}")
    print(f"{'='*60}")
    print(f"  Nodes: {n}, Edges: {m}, Greedy χ: {greedy_chi}")

    results = {
        "graph": graph_name,
        "n": n,
        "m": m,
        "greedy_chi": greedy_chi,
    }

    # HiGHS
    print(f"\n  Running HiGHS (time_limit={time_limit}s)...")
    chi, color_classes, solve_time, info = solve_coloring_ilp_highs(
        graph, max_colors=greedy_chi, time_limit=time_limit
    )

    if chi is not None:
        valid = validate_coloring(graph, color_classes)
        print(f"    χ={chi}, valid={valid}, time={solve_time:.2f}s")
        highs_result = {
            "chi": chi,
            "valid": valid,
            "time": solve_time,
            "color_classes": [list(cc) for cc in color_classes],
            **info,
        }
    else:
        print(f"    No solution found: {info.get('status', 'unknown')}")
        print(f"    Time: {solve_time:.2f}s")
        highs_result = {
            "chi": None,
            "time": solve_time,
            **info,
        }

    results["highs"] = highs_result
    save_result(graph_name, "highs", highs_result)

    # Hexaly (optional)
    if run_hexaly:
        print(f"\n  Running Hexaly (time_limit={time_limit}s)...")
        chi, color_classes, solve_time, info = solve_coloring_ilp_hexaly(
            graph, max_colors=greedy_chi, time_limit=time_limit
        )

        if chi is not None:
            valid = validate_coloring(graph, color_classes)
            gap_str = f", gap={info.get('gap', 0):.1%}" if info.get('gap') else ""
            print(f"    χ={chi}, valid={valid}, time={solve_time:.2f}s{gap_str}")
            hexaly_result = {
                "chi": chi,
                "valid": valid,
                "time": solve_time,
                "color_classes": [list(cc) for cc in color_classes],
                **info,
            }
        else:
            print(f"    No solution found: {info.get('status', 'unknown')}")
            print(f"    Time: {solve_time:.2f}s")
            hexaly_result = {
                "chi": None,
                "time": solve_time,
                **info,
            }

        results["hexaly"] = hexaly_result
        save_result(graph_name, "hexaly", hexaly_result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark exact ILP solvers for graph coloring"
    )
    parser.add_argument(
        "--graphs",
        nargs="*",
        help="Specific graphs to benchmark (e.g., er40_0.3 er50_0.3). "
             "If not specified, runs on all available benchmark graphs.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="Time limit in seconds per solver (default: 300)",
    )
    parser.add_argument(
        "--hexaly",
        action="store_true",
        help="Also run Hexaly solver (requires Hexaly installation)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmark graphs and exit",
    )

    args = parser.parse_args()

    available_graphs = list_benchmark_graphs()

    if args.list:
        print("Available benchmark graphs:")
        for g in available_graphs:
            print(f"  {g}")
        return

    if not available_graphs:
        print("No benchmark graphs found in benchmarks/")
        print("Run benchmark scripts first to generate graphs.")
        return

    # Select graphs to benchmark
    if args.graphs:
        graphs_to_run = []
        for g in args.graphs:
            if g in available_graphs:
                graphs_to_run.append(g)
            else:
                print(f"Warning: Graph '{g}' not found, skipping")
        if not graphs_to_run:
            print("No valid graphs specified")
            return
    else:
        graphs_to_run = available_graphs

    print(f"Benchmarking {len(graphs_to_run)} graph(s)")
    print(f"Time limit: {args.time_limit}s per solver")
    print(f"Solvers: HiGHS" + (", Hexaly" if args.hexaly else ""))

    all_results = []

    for graph_name in graphs_to_run:
        graph = load_benchmark_graph(graph_name)
        result = run_benchmark(graph_name, graph, args.time_limit, args.hexaly)
        all_results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    header = "| Graph            | Greedy | HiGHS χ | HiGHS Time |"
    if args.hexaly:
        header += " Hexaly χ | Hexaly Time | Gap   |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")

    for r in all_results:
        row = f"| {r['graph']:<16} | {r['greedy_chi']:>6} |"

        highs = r.get("highs", {})
        chi_h = highs.get("chi", "-")
        time_h = highs.get("time", 0)
        row += f" {str(chi_h):>7} | {time_h:>10.2f} |"

        if args.hexaly:
            hexaly = r.get("hexaly", {})
            chi_x = hexaly.get("chi", "-")
            time_x = hexaly.get("time", 0)
            gap = hexaly.get("gap")
            gap_str = f"{gap:.1%}" if gap else "-"
            row += f" {str(chi_x):>8} | {time_x:>11.2f} | {gap_str:>5} |"

        print(row)

    print("\nDone!")


if __name__ == "__main__":
    main()
