#!/usr/bin/env python
"""Full benchmark script with organized output for reproducibility.

Runs Greedy, LP, and optionally Dirac on ER graphs and saves complete
results to the benchmarks/ folder in a structured format.

Example usage:
    # Classical only (fast)
    uv run python scripts/run_benchmark_full.py --graphs 40 50 --classical-only

    # With Dirac (requires QCI_TOKEN)
    source ~/.zshrc
    uv run python scripts/run_benchmark_full.py --graphs 40 50

    # Larger graphs
    uv run python scripts/run_benchmark_full.py --graphs 75 100 --edge-prob 0.3
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, FrozenSet

import networkx as nx
import numpy as np

from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.graphs import erdos_renyi


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TimingInfo:
    """Per-oracle timing breakdown."""
    num_api_calls: int = 0
    total_api_seconds: float = 0.0
    total_extract_seconds: float = 0.0
    avg_api_seconds: float = 0.0
    avg_columns_per_call: float = 0.0
    total_columns_found: int = 0


@dataclass
class OracleConfig:
    """Configuration for reproducibility."""
    method: Optional[str] = None
    num_samples: Optional[int] = None
    multi_prune: Optional[bool] = None
    randomized_rounding: Optional[bool] = None
    num_random_rounds: Optional[int] = None
    random_seed: Optional[int] = None
    support_thresholds: Optional[List[float]] = None
    local_search_passes: Optional[int] = None


@dataclass
class MethodResult:
    """Result for a single method on a graph."""
    method: str
    graph_name: str
    n_nodes: int
    n_edges: int
    chi: Optional[int]
    color_classes: List[List[int]]
    valid: bool
    iterations: Optional[int] = None
    columns_found: Optional[int] = None
    wall_seconds: float = 0.0
    timing: Optional[TimingInfo] = None
    oracle_config: Optional[OracleConfig] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class GraphData:
    """Graph structure for reproducibility."""
    graph_name: str
    n_nodes: int
    n_edges: int
    seed: int
    edges: List[List[int]]


@dataclass
class Summary:
    """Combined summary for a graph."""
    graph_name: str
    n_nodes: int
    n_edges: int
    methods: Dict[str, Dict[str, Any]]
    best_chi: int
    winner: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_greedy(graph: nx.Graph, graph_name: str) -> MethodResult:
    """Run networkx greedy coloring."""
    t0 = time.time()
    coloring_dict = nx.coloring.greedy_color(graph, strategy="largest_first")
    elapsed = time.time() - t0

    # Convert to color classes
    color_classes: Dict[int, List[int]] = {}
    for node, color in coloring_dict.items():
        color_classes.setdefault(color, []).append(node)

    classes_list = [sorted(nodes) for nodes in color_classes.values()]
    chi = max(coloring_dict.values()) + 1 if coloring_dict else 0

    # Validate
    all_vertices = set(graph.nodes())
    covered = set()
    valid = True
    for nodes in classes_list:
        for v in nodes:
            if v in covered:
                valid = False
                break
            covered.add(v)
        for i, u in enumerate(nodes):
            for w in nodes[i + 1:]:
                if graph.has_edge(u, w):
                    valid = False
                    break

    if covered != all_vertices:
        valid = False

    return MethodResult(
        method="greedy",
        graph_name=graph_name,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        chi=chi,
        color_classes=classes_list,
        valid=valid,
        wall_seconds=round(elapsed, 4),
    )


def run_lp(
    graph: nx.Graph,
    graph_name: str,
    max_iterations: int = 500,
    verbose: bool = False,
) -> MethodResult:
    """Run LP relaxation oracle column generation."""
    from quantum_colgen.pricing.classical_lp import ClassicalLPPricingOracle

    oracle = ClassicalLPPricingOracle(
        multi_prune=True,
        randomized_rounding=True,
        num_random_rounds=10,
        random_seed=42,
    )

    t0 = time.time()
    num_colors, coloring, stats = column_generation(
        graph, oracle, max_iterations=max_iterations, verbose=verbose,
    )
    elapsed = time.time() - t0

    # Convert frozensets to lists
    classes_list = [sorted(list(c)) for c in coloring] if coloring else []
    valid = verify_coloring(graph, coloring) if coloring else False

    # Get timing info
    timing_summary = oracle.timer.summary()
    timing = TimingInfo(**timing_summary)

    oracle_config = OracleConfig(
        multi_prune=True,
        randomized_rounding=True,
        num_random_rounds=10,
        random_seed=42,
        local_search_passes=5,
    )

    return MethodResult(
        method="lp",
        graph_name=graph_name,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        chi=num_colors,
        color_classes=classes_list,
        valid=valid,
        iterations=stats.get("iterations", 0),
        columns_found=stats.get("columns_generated", 0),
        wall_seconds=round(elapsed, 2),
        timing=timing,
        oracle_config=oracle_config,
    )


def run_dirac(
    graph: nx.Graph,
    graph_name: str,
    max_iterations: int = 500,
    num_samples: int = 100,
    verbose: bool = False,
) -> MethodResult:
    """Run Dirac oracle column generation."""
    from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle

    oracle = DiracPricingOracle(
        method="gibbons",
        num_samples=num_samples,
        multi_prune=True,
        randomized_rounding=True,
        num_random_rounds=10,
        random_seed=42,
    )

    t0 = time.time()
    num_colors, coloring, stats = column_generation(
        graph, oracle, max_iterations=max_iterations, verbose=verbose,
    )
    elapsed = time.time() - t0

    classes_list = [sorted(list(c)) for c in coloring] if coloring else []
    valid = verify_coloring(graph, coloring) if coloring else False

    timing_summary = oracle.timer.summary()
    timing = TimingInfo(**timing_summary)

    oracle_config = OracleConfig(
        method="gibbons",
        num_samples=num_samples,
        multi_prune=True,
        randomized_rounding=True,
        num_random_rounds=10,
        random_seed=42,
    )

    return MethodResult(
        method="dirac",
        graph_name=graph_name,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        chi=num_colors,
        color_classes=classes_list,
        valid=valid,
        iterations=stats.get("iterations", 0),
        columns_found=stats.get("columns_generated", 0),
        wall_seconds=round(elapsed, 2),
        timing=timing,
        oracle_config=oracle_config,
    )


def save_result(result: MethodResult, filepath: Path) -> None:
    """Save a method result to JSON."""
    data = asdict(result)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {filepath}")


def save_graph(graph: nx.Graph, graph_name: str, seed: int, filepath: Path) -> None:
    """Save graph structure to JSON."""
    edges = [[int(u), int(v)] for u, v in sorted(graph.edges())]
    data = GraphData(
        graph_name=graph_name,
        n_nodes=graph.number_of_nodes(),
        n_edges=graph.number_of_edges(),
        seed=seed,
        edges=edges,
    )
    with open(filepath, "w") as f:
        json.dump(asdict(data), f, indent=2)
    print(f"  Saved: {filepath}")


def save_summary(results: Dict[str, MethodResult], graph_name: str, n_nodes: int, n_edges: int, filepath: Path) -> None:
    """Save combined summary for a graph."""
    methods = {}
    for method, result in results.items():
        methods[method] = {
            "chi": result.chi,
            "valid": result.valid,
            "wall_seconds": result.wall_seconds,
        }
        if result.iterations is not None:
            methods[method]["iterations"] = result.iterations

    chis = [r.chi for r in results.values() if r.chi is not None]
    best_chi = min(chis) if chis else 0
    winners = [m for m, r in results.items() if r.chi == best_chi]
    winner = "/".join(winners)

    summary = Summary(
        graph_name=graph_name,
        n_nodes=n_nodes,
        n_edges=n_edges,
        methods=methods,
        best_chi=best_chi,
        winner=winner,
    )

    with open(filepath, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"  Saved: {filepath}")


def generate_slides_data(output_dir: Path, graphs: List[str]) -> None:
    """Generate aggregated slides data from benchmark results."""
    slides_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graphs": {},
    }

    for graph_name in graphs:
        folder_name = graph_name.lower().replace("(", "").replace(")", "_").replace(",", "_").rstrip("_")
        graph_dir = output_dir / folder_name

        summary_path = graph_dir / "summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        # Load colorings from each method
        graph_data = {
            "graph_name": summary["graph_name"],
            "n_nodes": summary["n_nodes"],
            "n_edges": summary["n_edges"],
            "best_chi": summary["best_chi"],
            "winner": summary["winner"],
            "methods": {},
        }

        for method in ["greedy", "lp", "dirac"]:
            method_path = graph_dir / f"{method}.json"
            if method_path.exists():
                with open(method_path) as f:
                    method_data = json.load(f)
                graph_data["methods"][method] = {
                    "chi": method_data["chi"],
                    "color_classes": method_data["color_classes"],
                    "valid": method_data["valid"],
                    "wall_seconds": method_data["wall_seconds"],
                }
                if "iterations" in method_data:
                    graph_data["methods"][method]["iterations"] = method_data["iterations"]

        slides_data["graphs"][graph_name] = graph_data

    slides_path = output_dir / "slides_data.json"
    with open(slides_path, "w") as f:
        json.dump(slides_data, f, indent=2)
    print(f"\nSlides data saved: {slides_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run full benchmarks with organized output"
    )
    parser.add_argument(
        "--graphs", type=int, nargs="+", default=[40, 50],
        help="Node counts for ER graphs (default: 40 50)"
    )
    parser.add_argument(
        "--edge-prob", type=float, default=0.3,
        help="Edge probability (default: 0.3)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument(
        "--output-dir", type=str, default="benchmarks",
        help="Output directory (default: benchmarks)"
    )
    parser.add_argument("--classical-only", action="store_true", help="Skip Dirac")
    parser.add_argument("--dirac-only", action="store_true", help="Only run Dirac")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    graph_names = []

    for n in args.graphs:
        graph_name = f"ER({n},{args.edge_prob})"
        graph_names.append(graph_name)

        folder_name = f"er{n}_{args.edge_prob}"
        graph_dir = output_dir / folder_name
        graph_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {graph_name}")
        print(f"{'='*60}")

        # Generate graph
        G = erdos_renyi(n, args.edge_prob, seed=args.seed)
        print(f"  Generated: n={G.number_of_nodes()}, m={G.number_of_edges()}")

        # Save graph structure
        save_graph(G, graph_name, args.seed, graph_dir / "graph.json")

        results: Dict[str, MethodResult] = {}

        # Greedy
        if not args.dirac_only:
            print(f"\n  Running Greedy...")
            greedy_result = run_greedy(G, graph_name)
            results["greedy"] = greedy_result
            print(f"    chi={greedy_result.chi}, valid={greedy_result.valid}, time={greedy_result.wall_seconds}s")
            save_result(greedy_result, graph_dir / "greedy.json")

        # LP
        if not args.dirac_only:
            print(f"\n  Running LP CG...")
            lp_result = run_lp(G, graph_name, max_iterations=args.max_iterations, verbose=args.verbose)
            results["lp"] = lp_result
            t = lp_result.timing
            timing_str = ""
            if t and t.num_api_calls > 0:
                timing_str = f", api_calls={t.num_api_calls}, avg_cols/call={t.avg_columns_per_call:.1f}"
            print(f"    chi={lp_result.chi}, valid={lp_result.valid}, iters={lp_result.iterations}, time={lp_result.wall_seconds}s{timing_str}")
            save_result(lp_result, graph_dir / "lp.json")

        # Dirac
        if not args.classical_only:
            print(f"\n  Running Dirac CG...")
            dirac_result = run_dirac(
                G, graph_name,
                max_iterations=args.max_iterations,
                num_samples=args.num_samples,
                verbose=args.verbose,
            )
            results["dirac"] = dirac_result
            t = dirac_result.timing
            timing_str = ""
            if t and t.num_api_calls > 0:
                timing_str = f", api_calls={t.num_api_calls}, avg_cols/call={t.avg_columns_per_call:.1f}"
            print(f"    chi={dirac_result.chi}, valid={dirac_result.valid}, iters={dirac_result.iterations}, time={dirac_result.wall_seconds}s{timing_str}")
            save_result(dirac_result, graph_dir / "dirac.json")

        # Save summary
        save_summary(results, graph_name, G.number_of_nodes(), G.number_of_edges(), graph_dir / "summary.json")

    # Generate slides data
    generate_slides_data(output_dir, graph_names)

    # Print final summary table
    print(f"\n\n{'='*80}")
    print("Summary")
    print("="*80)
    print(f"| {'Graph':<16} | {'Greedy':>6} | {'LP':>6} | {'Dirac':>6} | {'Winner':>12} |")
    print(f"|{'-'*18}|{'-'*8}|{'-'*8}|{'-'*8}|{'-'*14}|")

    for graph_name in graph_names:
        folder_name = f"er{graph_name.split('(')[1].split(',')[0]}_{args.edge_prob}"
        summary_path = output_dir / folder_name / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            greedy = summary["methods"].get("greedy", {}).get("chi", "-")
            lp = summary["methods"].get("lp", {}).get("chi", "-")
            dirac = summary["methods"].get("dirac", {}).get("chi", "-")
            winner = summary["winner"]
            print(f"| {graph_name:<16} | {greedy:>6} | {lp:>6} | {dirac:>6} | {winner:>12} |")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
