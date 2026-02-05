#!/usr/bin/env python
"""Generate graph colorings for ER(40,0.3) and ER(50,0.3) slides.

This script generates reproducible colorings using:
- Greedy coloring (networkx, deterministic)
- LP-based column generation (fast classical baseline)
- Dirac column generation (requires QCI_TOKEN)

Outputs:
- manim/src/quantum_colgen_slides/data/graph_colorings.json (for slides)
- results/coloring_solutions_er40.json (raw output for reproducibility)
- results/coloring_solutions_er50.json (raw output for reproducibility)

Usage:
    # Full run with Dirac (requires QCI_TOKEN)
    source ~/.zshrc
    uv run python manim/scripts/generate_colorings.py

    # Classical only (Greedy + LP, no Dirac API calls)
    uv run python manim/scripts/generate_colorings.py --classical-only

    # Specific graphs
    uv run python manim/scripts/generate_colorings.py --graphs ER40

    # Verbose output
    uv run python manim/scripts/generate_colorings.py --verbose
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from quantum_colgen.column_generation import column_generation, validate_coloring
from quantum_colgen.pricing.classical_lp import ClassicalLPPricingOracle
from quantum_colgen.graphs import erdos_renyi


def greedy_coloring(G: nx.Graph) -> Tuple[int, List[Set[int]]]:
    """Run networkx greedy coloring with largest_first strategy.

    Returns (chi, color_classes) where color_classes is a list of sets.
    """
    coloring_dict = nx.coloring.greedy_color(G, strategy="largest_first")
    if not coloring_dict:
        return 0, []

    chi = max(coloring_dict.values()) + 1
    color_classes: List[Set[int]] = [set() for _ in range(chi)]
    for node, color_idx in coloring_dict.items():
        color_classes[color_idx].add(node)

    return chi, color_classes


def run_lp_coloring(
    G: nx.Graph,
    max_iterations: int = 500,
    verbose: bool = False,
) -> Tuple[int, List[Set[int]], Dict]:
    """Run LP-based column generation.

    Returns (chi, color_classes, stats).
    """
    oracle = ClassicalLPPricingOracle()
    chi, coloring, stats = column_generation(
        G, oracle, max_iterations=max_iterations, verbose=verbose
    )
    return chi, coloring if coloring else [], stats


def run_dirac_coloring(
    G: nx.Graph,
    max_iterations: int = 500,
    num_samples: int = 100,
    verbose: bool = False,
) -> Tuple[int, List[Set[int]], Dict]:
    """Run Dirac-based column generation.

    Requires QCI_TOKEN environment variable.
    Returns (chi, color_classes, stats).
    """
    # Import here to avoid import errors when Dirac not available
    from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle

    oracle = DiracPricingOracle(
        method="gibbons",
        num_samples=num_samples,
        multi_prune=True,
        randomized_rounding=True,
        num_random_rounds=10,
        random_seed=42,
    )
    chi, coloring, stats = column_generation(
        G, oracle, max_iterations=max_iterations, verbose=verbose
    )
    return chi, coloring if coloring else [], stats


def generate_graph_data(
    name: str,
    n: int,
    p: float,
    seed: int = 42,
    run_dirac: bool = True,
    max_iterations: int = 500,
    verbose: bool = False,
) -> Dict:
    """Generate all colorings for a single graph.

    Returns dict with graph info and colorings for each method.
    """
    print(f"\n{'='*60}")
    print(f"Generating {name}: n={n}, p={p}, seed={seed}")
    print(f"{'='*60}")

    G = erdos_renyi(n, p, seed=seed)
    n_edges = G.number_of_edges()
    edges = list(G.edges())

    print(f"Graph has {n} nodes, {n_edges} edges")

    result = {
        "graph_name": name,
        "n_nodes": n,
        "n_edges": n_edges,
        "seed": seed,
        "edges": edges,
        "methods": {},
    }

    # Greedy coloring
    print("\n--- Greedy coloring ---")
    t0 = time.time()
    greedy_chi, greedy_classes = greedy_coloring(G)
    greedy_time = time.time() - t0

    # Validate greedy
    vr = validate_coloring(G, greedy_classes)
    print(f"Greedy: chi={greedy_chi}, valid={vr.valid}, time={greedy_time:.3f}s")

    result["methods"]["greedy"] = {
        "chi": greedy_chi,
        "color_classes": [sorted(list(c)) for c in greedy_classes],
        "valid": vr.valid,
        "wall_seconds": round(greedy_time, 4),
    }

    # LP coloring
    print("\n--- LP-based CG ---")
    t0 = time.time()
    lp_chi, lp_classes, lp_stats = run_lp_coloring(G, max_iterations, verbose)
    lp_time = time.time() - t0

    if lp_classes:
        vr = validate_coloring(G, lp_classes)
        print(f"LP: chi={lp_chi}, valid={vr.valid}, iterations={lp_stats.get('iterations', 0)}, time={lp_time:.2f}s")
        result["methods"]["lp"] = {
            "chi": lp_chi,
            "color_classes": [sorted(list(c)) for c in lp_classes],
            "valid": vr.valid,
            "iterations": lp_stats.get("iterations", 0),
            "columns_found": lp_stats.get("columns_found", 0),
            "wall_seconds": round(lp_time, 2),
        }
    else:
        print(f"LP: FAILED")
        result["methods"]["lp"] = {"chi": None, "color_classes": [], "valid": False}

    # Dirac coloring (optional)
    if run_dirac:
        print("\n--- Dirac-based CG ---")
        if not os.environ.get("QCI_TOKEN"):
            print("WARNING: QCI_TOKEN not set, skipping Dirac")
            result["methods"]["dirac"] = {"chi": None, "color_classes": [], "valid": False, "error": "QCI_TOKEN not set"}
        else:
            t0 = time.time()
            try:
                dirac_chi, dirac_classes, dirac_stats = run_dirac_coloring(
                    G, max_iterations, verbose=verbose
                )
                dirac_time = time.time() - t0

                if dirac_classes:
                    vr = validate_coloring(G, dirac_classes)
                    print(f"Dirac: chi={dirac_chi}, valid={vr.valid}, iterations={dirac_stats.get('iterations', 0)}, time={dirac_time:.2f}s")
                    result["methods"]["dirac"] = {
                        "chi": dirac_chi,
                        "color_classes": [sorted(list(c)) for c in dirac_classes],
                        "valid": vr.valid,
                        "iterations": dirac_stats.get("iterations", 0),
                        "columns_found": dirac_stats.get("columns_found", 0),
                        "wall_seconds": round(dirac_time, 2),
                    }
                else:
                    print(f"Dirac: FAILED")
                    result["methods"]["dirac"] = {"chi": None, "color_classes": [], "valid": False}
            except Exception as e:
                print(f"Dirac: ERROR - {e}")
                result["methods"]["dirac"] = {"chi": None, "color_classes": [], "valid": False, "error": str(e)}

    return result


def build_slides_data(results: List[Dict]) -> Dict:
    """Build simplified data structure for manim slides."""
    slides_data = {}

    for r in results:
        name = r["graph_name"]
        slides_data[name] = {
            "n_nodes": r["n_nodes"],
            "n_edges": r["n_edges"],
            "edges": r["edges"],
            "colorings": {},
        }

        for method_name, method_data in r["methods"].items():
            if method_data.get("chi") is not None:
                slides_data[name]["colorings"][method_name] = {
                    "chi": method_data["chi"],
                    "color_classes": method_data["color_classes"],
                }

    return slides_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate graph colorings for presentation slides",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=["ER40", "ER50", "all"],
        default=["all"],
        help="Which graphs to generate (default: all)",
    )
    parser.add_argument(
        "--classical-only",
        action="store_true",
        help="Skip Dirac oracle (no QCI API calls)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Max CG iterations (default: 500)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output during CG",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: auto-detect)",
    )

    args = parser.parse_args()

    # Determine output paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "results"

    slides_data_path = project_root / "manim" / "src" / "quantum_colgen_slides" / "data" / "graph_colorings.json"

    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    slides_data_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine which graphs to run
    graphs_to_run = []
    if "all" in args.graphs or "ER40" in args.graphs:
        graphs_to_run.append(("ER(40,0.3)", 40, 0.3))
    if "all" in args.graphs or "ER50" in args.graphs:
        graphs_to_run.append(("ER(50,0.3)", 50, 0.3))

    run_dirac = not args.classical_only

    print(f"Generating colorings for: {[g[0] for g in graphs_to_run]}")
    print(f"Run Dirac: {run_dirac}")
    print(f"Max iterations: {args.max_iterations}")

    all_results = []

    for name, n, p in graphs_to_run:
        result = generate_graph_data(
            name, n, p,
            seed=42,
            run_dirac=run_dirac,
            max_iterations=args.max_iterations,
            verbose=args.verbose,
        )
        all_results.append(result)

        # Save individual raw result
        raw_path = output_dir / f"coloring_solutions_{name.lower().replace('(', '').replace(')', '').replace(',', '_')}.json"
        with open(raw_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nRaw output saved to: {raw_path}")

    # Build and save slides data
    slides_data = build_slides_data(all_results)
    with open(slides_data_path, "w") as f:
        json.dump(slides_data, f, indent=2)
    print(f"\nSlides data saved to: {slides_data_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n{r['graph_name']} ({r['n_nodes']} nodes, {r['n_edges']} edges):")
        for method, data in r["methods"].items():
            chi = data.get("chi", "FAIL")
            valid = data.get("valid", False)
            status = "valid" if valid else "INVALID"
            print(f"  {method:8s}: chi={chi}, {status}")


if __name__ == "__main__":
    main()
