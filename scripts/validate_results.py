#!/usr/bin/env python
"""Validate coloring solutions by re-running CG and checking each result.

Re-generates the benchmark graphs (same seeds), runs CG with the specified
oracle, and performs detailed validation of every coloring.  Optionally saves
the full coloring assignments to a JSON file for offline inspection.

Usage:
    # Validate classical CG on predefined + small ER graphs
    uv run python scripts/validate_results.py

    # Validate Dirac oracle
    uv run python scripts/validate_results.py --oracle dirac

    # Save coloring solutions to file
    uv run python scripts/validate_results.py --save-colorings results/colorings.json

    # Validate specific ER sizes
    uv run python scripts/validate_results.py --er-sizes 8 10 12

    # Validate from a previously saved JSON benchmark file
    uv run python scripts/validate_results.py --from-json results/benchmark_01.json
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from quantum_colgen.column_generation import (
    column_generation,
    validate_coloring,
    verify_coloring,
    ValidationResult,
)
from quantum_colgen.pricing.classical import ClassicalPricingOracle
from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle
from quantum_colgen.graphs import TEST_GRAPHS, KNOWN_CHROMATIC, erdos_renyi


# ---------------------------------------------------------------------------
# Graph catalogue (same as benchmark.py)
# ---------------------------------------------------------------------------

def build_graph_catalogue(
    er_sizes: List[int],
    er_probs: List[float],
    seed: int = 42,
) -> List[Tuple[str, nx.Graph, Optional[int]]]:
    """Return (name, graph, expected_chi_or_None) tuples."""
    catalogue = []
    for name, factory in TEST_GRAPHS.items():
        catalogue.append((name, factory(), KNOWN_CHROMATIC[name]))
    for n in er_sizes:
        for p in er_probs:
            G = erdos_renyi(n, p, seed=seed)
            if G.number_of_edges() == 0:
                continue
            catalogue.append((f"ER({n},{p})", G, None))
    return catalogue


def _nx_greedy_chi(G: nx.Graph) -> int:
    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    return max(coloring.values()) + 1 if coloring else 0


# ---------------------------------------------------------------------------
# Validation from a saved benchmark JSON
# ---------------------------------------------------------------------------

def validate_from_json(json_path: str, seed: int = 42) -> bool:
    """Load benchmark results JSON, regenerate graphs, re-run CG, validate.

    Returns True if all entries pass validation.
    """
    with open(json_path) as f:
        records = json.load(f)

    # Build a lookup of graphs by name
    graph_cache: Dict[str, nx.Graph] = {}
    for name, factory in TEST_GRAPHS.items():
        graph_cache[name] = factory()

    all_pass = True
    for i, rec in enumerate(records):
        gname = rec["graph_name"]
        oracle_name = rec["oracle"]
        n_nodes = rec["n_nodes"]
        chi_reported = rec["chi"]
        valid_reported = rec["valid"]

        # Regenerate graph if not cached
        if gname not in graph_cache:
            if gname.startswith("ER("):
                # Parse ER(n,p)
                inner = gname[3:-1]
                parts = inner.split(",")
                n, p = int(parts[0]), float(parts[1])
                graph_cache[gname] = erdos_renyi(n, p, seed=seed)
            else:
                print(f"  [{i+1}] {gname} ({oracle_name}): SKIP (unknown graph)")
                continue

        G = graph_cache[gname]

        # Verify graph matches expected size
        if G.number_of_nodes() != n_nodes:
            print(f"  [{i+1}] {gname} ({oracle_name}): GRAPH MISMATCH "
                  f"(expected {n_nodes} nodes, got {G.number_of_nodes()})")
            all_pass = False
            continue

        # Re-run CG
        if oracle_name == "classical":
            oracle = ClassicalPricingOracle()
        elif oracle_name == "dirac":
            oracle = DiracPricingOracle()
        else:
            print(f"  [{i+1}] {gname} ({oracle_name}): SKIP (unknown oracle)")
            continue

        max_iter = rec.get("max_iterations", 500)
        num_colors, coloring, stats = column_generation(
            G, oracle, max_iterations=max_iter,
        )

        if not coloring:
            print(f"  [{i+1}] {gname} ({oracle_name}): CG FAILED (no coloring)")
            all_pass = False
            continue

        # Detailed validation
        result = validate_coloring(G, coloring)
        status = "PASS" if result.valid else "FAIL"
        print(f"  [{i+1}] {gname} ({oracle_name}): chi={num_colors}  [{status}]", end="")

        if not result.valid:
            all_pass = False
            _print_errors(result)
        else:
            print()

    return all_pass


# ---------------------------------------------------------------------------
# Fresh validation run
# ---------------------------------------------------------------------------

def validate_fresh(
    catalogue: List[Tuple[str, nx.Graph, Optional[int]]],
    oracle_name: str,
    dirac_kwargs: dict,
    max_iterations: int = 500,
    save_path: Optional[str] = None,
    verbose: bool = False,
) -> bool:
    """Run CG on each graph and validate. Returns True if all pass."""
    all_pass = True
    total = len(catalogue)
    saved_results = []

    for idx, (gname, G, expected_chi) in enumerate(catalogue, 1):
        print(f"\n{'='*60}")
        print(f"  [{idx}/{total}] {gname} ({oracle_name})  n={G.number_of_nodes()} m={G.number_of_edges()}")
        print(f"{'='*60}")

        if oracle_name == "classical":
            oracle = ClassicalPricingOracle()
        else:
            oracle = DiracPricingOracle(**dirac_kwargs)

        t0 = time.time()
        num_colors, coloring, stats = column_generation(
            G, oracle, max_iterations=max_iterations, verbose=verbose,
        )
        elapsed = round(time.time() - t0, 2)

        if not coloring:
            print(f"  CG FAILED — no coloring produced")
            all_pass = False
            saved_results.append(_make_record(
                gname, oracle_name, G, expected_chi, None, [], None, elapsed,
            ))
            continue

        # Detailed validation
        vr = validate_coloring(G, coloring)
        greedy = _nx_greedy_chi(G)

        # Print results
        status = "PASS" if vr.valid else "FAIL"
        print(f"  chi = {num_colors}  (expected={expected_chi}, greedy={greedy})")
        print(f"  colors used = {vr.num_colors}")
        print(f"  vertices covered = {vr.num_vertices_covered}/{vr.num_vertices_expected}")
        print(f"  time = {elapsed}s  iters = {stats.get('iterations', 0)}")

        if not vr.valid:
            all_pass = False
            _print_errors(vr)
            print(f"  [{status}]")
        else:
            # Additional checks
            warnings = []
            if expected_chi is not None and num_colors != expected_chi:
                warnings.append(f"chi={num_colors} != expected={expected_chi}")
            if num_colors > greedy:
                warnings.append(f"chi={num_colors} > greedy={greedy}")
            if warnings:
                print(f"  WARNINGS: {'; '.join(warnings)}")
            print(f"  [{status}]")

        # Print color classes
        if verbose:
            for c_idx, cset in enumerate(coloring):
                print(f"    color {c_idx}: {sorted(cset)}")

        # Save record
        saved_results.append(_make_record(
            gname, oracle_name, G, expected_chi, num_colors,
            coloring, vr, elapsed,
        ))

    # Save colorings if requested
    if save_path and saved_results:
        with open(save_path, "w") as f:
            json.dump(saved_results, f, indent=2)
        print(f"\nColoring solutions saved to {save_path}")

    return all_pass


def _make_record(
    gname, oracle_name, G, expected_chi, num_colors, coloring, vr, elapsed,
):
    return {
        "graph_name": gname,
        "oracle": oracle_name,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "expected_chi": expected_chi,
        "chi": num_colors,
        "valid": vr.valid if vr else False,
        "wall_seconds": elapsed,
        "color_classes": [sorted(list(c)) for c in coloring] if coloring else [],
        "num_colors": vr.num_colors if vr else 0,
        "vertices_covered": vr.num_vertices_covered if vr else 0,
        "missing_vertices": vr.missing_vertices if vr else [],
        "duplicate_vertices": vr.duplicate_vertices if vr else [],
        "edge_violations": [
            {"color_idx": ci, "u": u, "v": v}
            for ci, u, v in (vr.edge_violations if vr else [])
        ],
    }


def _print_errors(vr: ValidationResult):
    if vr.missing_vertices:
        print(f"\n    MISSING VERTICES: {vr.missing_vertices}")
    if vr.duplicate_vertices:
        print(f"\n    DUPLICATE VERTICES: {vr.duplicate_vertices}")
    if vr.edge_violations:
        print(f"\n    EDGE VIOLATIONS ({len(vr.edge_violations)}):")
        for ci, u, v in vr.edge_violations[:10]:
            print(f"      color {ci}: edge ({u}, {v})")
        if len(vr.edge_violations) > 10:
            print(f"      ... and {len(vr.edge_violations) - 10} more")
    if vr.empty_color_classes:
        print(f"\n    EMPTY COLOR CLASSES: {vr.empty_color_classes}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(json_path: str):
    """Print a summary table from a saved colorings JSON."""
    with open(json_path) as f:
        records = json.load(f)

    print(f"\n{'='*80}")
    print(f"Validation Summary — {json_path}")
    print(f"{'='*80}")
    print(
        f"| {'Graph':<16} | {'Oracle':<10} | {'n':>3} | {'m':>3} "
        f"| {'chi':>3} | {'exp':>3} | {'valid':>5} | {'violations':>10} "
        f"| {'missing':>7} | {'dupes':>5} |"
    )
    print(
        f"|{'-'*18}|{'-'*12}|{'-'*5}|{'-'*5}"
        f"|{'-'*5}|{'-'*5}|{'-'*7}|{'-'*12}"
        f"|{'-'*9}|{'-'*7}|"
    )

    n_pass = 0
    n_fail = 0
    for r in records:
        exp = str(r["expected_chi"]) if r["expected_chi"] is not None else "-"
        chi = str(r["chi"]) if r["chi"] is not None else "FAIL"
        violations = len(r.get("edge_violations", []))
        missing = len(r.get("missing_vertices", []))
        dupes = len(r.get("duplicate_vertices", []))
        valid_str = "True" if r["valid"] else "FALSE"
        if r["valid"]:
            n_pass += 1
        else:
            n_fail += 1
        print(
            f"| {r['graph_name']:<16} | {r['oracle']:<10} | {r['n_nodes']:>3} | {r['n_edges']:>3} "
            f"| {chi:>3} | {exp:>3} | {valid_str:>5} | {violations:>10} "
            f"| {missing:>7} | {dupes:>5} |"
        )

    total = n_pass + n_fail
    print(f"\nTotal: {total}  Pass: {n_pass}  Fail: {n_fail}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate coloring solutions from CG",
    )
    parser.add_argument(
        "--oracle", default="classical", choices=["classical", "dirac"],
        help="Pricing oracle to use (default: classical)",
    )
    parser.add_argument(
        "--er-sizes", type=int, nargs="+", default=[8, 10, 12],
        help="Erdos-Renyi node counts (default: 8 10 12)",
    )
    parser.add_argument(
        "--er-probs", type=float, nargs="+", default=[0.3, 0.5],
        help="Erdos-Renyi edge probabilities (default: 0.3 0.5)",
    )
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--relax-schedule", type=int, default=2)
    parser.add_argument("--method", default="gibbons", choices=["gibbons", "filter"])
    parser.add_argument(
        "--save-colorings", type=str, default=None,
        help="Save full coloring solutions to JSON file",
    )
    parser.add_argument(
        "--from-json", type=str, default=None,
        help="Validate by re-running graphs from a benchmark JSON file",
    )
    parser.add_argument(
        "--summary", type=str, default=None,
        help="Print summary from a saved colorings JSON file (no re-run)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Summary mode: just print from saved file
    if args.summary:
        print_summary(args.summary)
        return 0

    # Validation from saved benchmark JSON
    if args.from_json:
        print(f"Validating results from {args.from_json}")
        print(f"Oracle: {args.oracle}  (re-running CG to validate)")
        ok = validate_from_json(args.from_json)
        print(f"\n{'='*60}")
        print(f"Overall: {'ALL PASS' if ok else 'FAILURES DETECTED'}")
        return 0 if ok else 1

    # Fresh validation run
    dirac_kwargs = {
        "method": args.method,
        "num_samples": args.num_samples,
        "relax_schedule": args.relax_schedule,
    }

    catalogue = build_graph_catalogue(args.er_sizes, args.er_probs)

    print(f"Validation run: oracle={args.oracle}  max_iterations={args.max_iterations}")
    print(f"Graphs: {len(catalogue)} ({len(TEST_GRAPHS)} predefined + ER)")

    ok = validate_fresh(
        catalogue,
        args.oracle,
        dirac_kwargs,
        max_iterations=args.max_iterations,
        save_path=args.save_colorings,
        verbose=args.verbose,
    )

    print(f"\n{'='*60}")
    print(f"Overall: {'ALL PASS' if ok else 'FAILURES DETECTED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
