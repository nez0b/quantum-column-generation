#!/usr/bin/env python
"""Migrate existing results from results/ to benchmarks/ folder structure.

Reads results/coloring_solutions_er*.json and creates the structured
benchmarks/ output.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def migrate_results():
    """Migrate existing results to new structure."""
    results_dir = Path("results")
    benchmarks_dir = Path("benchmarks")
    benchmarks_dir.mkdir(exist_ok=True)

    # Files to migrate
    source_files = [
        ("results/coloring_solutions_er40_0.3.json", "er40_0.3"),
        ("results/coloring_solutions_er50_0.3.json", "er50_0.3"),
    ]

    graph_names = []

    for source_file, folder_name in source_files:
        source_path = Path(source_file)
        if not source_path.exists():
            print(f"Skip: {source_file} not found")
            continue

        print(f"\nMigrating: {source_file}")

        with open(source_path) as f:
            data = json.load(f)

        graph_name = data["graph_name"]
        n_nodes = data["n_nodes"]
        n_edges = data["n_edges"]
        seed = data.get("seed", 42)
        edges = data.get("edges", [])

        graph_names.append(graph_name)

        # Create folder
        graph_dir = benchmarks_dir / folder_name
        graph_dir.mkdir(exist_ok=True)

        # Save graph.json
        graph_data = {
            "graph_name": graph_name,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "seed": seed,
            "edges": edges,
        }
        with open(graph_dir / "graph.json", "w") as f:
            json.dump(graph_data, f, indent=2)
        print(f"  Saved: {graph_dir / 'graph.json'}")

        # Migrate each method
        methods_summary = {}

        for method, method_data in data.get("methods", {}).items():
            chi = method_data.get("chi")
            color_classes = method_data.get("color_classes", [])
            valid = method_data.get("valid", True)
            iterations = method_data.get("iterations")
            columns_found = method_data.get("columns_found")
            wall_seconds = method_data.get("wall_seconds", 0)

            method_result = {
                "method": method,
                "graph_name": graph_name,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "chi": chi,
                "color_classes": color_classes,
                "valid": valid,
                "wall_seconds": wall_seconds,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if iterations is not None:
                method_result["iterations"] = iterations
            if columns_found is not None:
                method_result["columns_found"] = columns_found

            # Add oracle config for dirac
            if method == "dirac":
                method_result["oracle_config"] = {
                    "method": "gibbons",
                    "num_samples": 100,
                    "multi_prune": True,
                    "randomized_rounding": True,
                    "num_random_rounds": 10,
                    "random_seed": 42,
                }
            elif method == "lp":
                method_result["oracle_config"] = {
                    "multi_prune": True,
                    "randomized_rounding": True,
                    "num_random_rounds": 10,
                    "random_seed": 42,
                    "local_search_passes": 5,
                }

            with open(graph_dir / f"{method}.json", "w") as f:
                json.dump(method_result, f, indent=2)
            print(f"  Saved: {graph_dir / f'{method}.json'}")

            methods_summary[method] = {
                "chi": chi,
                "valid": valid,
                "wall_seconds": wall_seconds,
            }
            if iterations is not None:
                methods_summary[method]["iterations"] = iterations

        # Create summary
        chis = [m["chi"] for m in methods_summary.values() if m.get("chi")]
        best_chi = min(chis) if chis else 0
        winners = [m for m, d in methods_summary.items() if d.get("chi") == best_chi]

        summary = {
            "graph_name": graph_name,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "methods": methods_summary,
            "best_chi": best_chi,
            "winner": "/".join(winners),
        }

        with open(graph_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {graph_dir / 'summary.json'}")

    # Generate slides_data.json
    slides_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graphs": {},
    }

    for source_file, folder_name in source_files:
        source_path = Path(source_file)
        if not source_path.exists():
            continue

        with open(source_path) as f:
            data = json.load(f)

        graph_name = data["graph_name"]
        n_nodes = data["n_nodes"]
        n_edges = data["n_edges"]

        graph_data = {
            "graph_name": graph_name,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "methods": {},
        }

        for method, method_data in data.get("methods", {}).items():
            graph_data["methods"][method] = {
                "chi": method_data.get("chi"),
                "color_classes": method_data.get("color_classes", []),
                "valid": method_data.get("valid", True),
                "wall_seconds": method_data.get("wall_seconds", 0),
            }
            if "iterations" in method_data:
                graph_data["methods"][method]["iterations"] = method_data["iterations"]

        chis = [m["chi"] for m in graph_data["methods"].values() if m.get("chi")]
        graph_data["best_chi"] = min(chis) if chis else 0
        winners = [m for m, d in graph_data["methods"].items() if d.get("chi") == graph_data["best_chi"]]
        graph_data["winner"] = "/".join(winners)

        slides_data["graphs"][graph_name] = graph_data

    with open(benchmarks_dir / "slides_data.json", "w") as f:
        json.dump(slides_data, f, indent=2)
    print(f"\nSlides data saved: {benchmarks_dir / 'slides_data.json'}")

    print("\nMigration complete!")


if __name__ == "__main__":
    migrate_results()
