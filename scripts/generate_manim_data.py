#!/usr/bin/env python
"""Generate manim slides data from benchmarks folder.

Reads benchmark results and creates the graph_colorings.json file
expected by the manim slides.

Usage:
    uv run python scripts/generate_manim_data.py
"""

import json
import sys
from pathlib import Path


def main():
    """Generate graph_colorings.json from benchmarks folder."""
    benchmarks_dir = Path("benchmarks")
    output_path = Path("manim/src/quantum_colgen_slides/data/graph_colorings.json")

    if not benchmarks_dir.exists():
        print(f"Error: {benchmarks_dir} does not exist")
        return 1

    output_data = {}

    # Find all graph directories
    for graph_dir in sorted(benchmarks_dir.iterdir()):
        if not graph_dir.is_dir():
            continue

        graph_json = graph_dir / "graph.json"
        if not graph_json.exists():
            continue

        # Load graph data
        with open(graph_json) as f:
            graph_data = json.load(f)

        graph_name = graph_data["graph_name"]
        print(f"Processing {graph_name}...")

        entry = {
            "n_nodes": graph_data["n_nodes"],
            "n_edges": graph_data["n_edges"],
            "edges": graph_data["edges"],
            "colorings": {},
        }

        # Load each method's coloring
        for method in ["greedy", "lp", "dirac"]:
            method_json = graph_dir / f"{method}.json"
            if method_json.exists():
                with open(method_json) as f:
                    method_data = json.load(f)
                entry["colorings"][method] = {
                    "chi": method_data["chi"],
                    "color_classes": method_data["color_classes"],
                    "valid": method_data["valid"],
                }

        output_data[graph_name] = entry

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nGenerated: {output_path}")
    print(f"Graphs included: {list(output_data.keys())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
