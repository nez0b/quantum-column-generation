#!/usr/bin/env python
"""Test if enhanced Dirac extraction yields better chi than greedy.

Compares:
1. Greedy coloring (networkx)
2. Baseline Dirac CG (current defaults)
3. Enhanced Dirac CG (multi_prune + randomized_rounding)

Usage:
    source ~/.zshrc
    uv run python scripts/test_enhanced_extraction_chi.py --nodes 30 --edge-prob 0.3
"""

import argparse
import sys
import time

import networkx as nx

from quantum_colgen.graphs import erdos_renyi
from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle, DIRAC_AVAILABLE


def run_greedy(graph: nx.Graph) -> int:
    """Run greedy coloring and return chi."""
    coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
    return max(coloring.values()) + 1 if coloring else 0


def run_dirac_cg(
    graph: nx.Graph,
    enhanced: bool = False,
    max_iterations: int = 500,
    verbose: bool = False,
) -> tuple:
    """Run Dirac CG and return (chi, valid, iterations, columns, time)."""
    if enhanced:
        oracle = DiracPricingOracle(
            method="gibbons",
            num_samples=100,
            relax_schedule=2,
            multi_prune=True,
            num_random_prune_trials=3,
            randomized_rounding=True,
            num_random_rounds=10,
            random_seed=42,
        )
    else:
        oracle = DiracPricingOracle(
            method="gibbons",
            num_samples=100,
            relax_schedule=2,
        )

    t0 = time.time()
    num_colors, coloring, stats = column_generation(
        graph, oracle, max_iterations=max_iterations, verbose=verbose,
    )
    elapsed = time.time() - t0

    valid = verify_coloring(graph, coloring) if coloring else False
    iters = stats.get("iterations", 0)
    cols = stats.get("columns_generated", 0)

    # Get timing breakdown
    timer = oracle.timer.summary()

    return num_colors, valid, iters, cols, elapsed, timer


def main():
    parser = argparse.ArgumentParser(
        description="Test if enhanced Dirac extraction yields better chi"
    )
    parser.add_argument("--nodes", "-n", type=int, default=30)
    parser.add_argument("--edge-prob", "-p", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if not DIRAC_AVAILABLE:
        print("ERROR: Dirac not available")
        return 1

    # Generate graph
    graph = erdos_renyi(args.nodes, args.edge_prob, seed=args.seed)
    print(f"\n{'='*70}")
    print(f"Graph: ER({args.nodes}, {args.edge_prob})")
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
    print(f"{'='*70}")

    # Greedy baseline
    print("\n[1/3] Running greedy coloring...")
    t0 = time.time()
    greedy_chi = run_greedy(graph)
    greedy_time = time.time() - t0
    print(f"  Greedy chi: {greedy_chi} (time: {greedy_time:.3f}s)")

    # Baseline Dirac CG
    print("\n[2/3] Running BASELINE Dirac CG...")
    chi_base, valid_base, iters_base, cols_base, time_base, timer_base = run_dirac_cg(
        graph, enhanced=False, max_iterations=args.max_iterations, verbose=args.verbose
    )
    valid_str = "VALID" if valid_base else "INVALID"
    print(f"  Baseline chi: {chi_base} [{valid_str}]")
    print(f"  Iterations: {iters_base}, Columns: {cols_base}, Time: {time_base:.1f}s")
    print(f"  API calls: {timer_base['num_api_calls']}, "
          f"Cols found: {timer_base['total_columns_found']}, "
          f"Avg cols/call: {timer_base['avg_columns_per_call']:.1f}")

    # Enhanced Dirac CG
    print("\n[3/3] Running ENHANCED Dirac CG (multi_prune + randomized_rounding)...")
    chi_enh, valid_enh, iters_enh, cols_enh, time_enh, timer_enh = run_dirac_cg(
        graph, enhanced=True, max_iterations=args.max_iterations, verbose=args.verbose
    )
    valid_str = "VALID" if valid_enh else "INVALID"
    print(f"  Enhanced chi: {chi_enh} [{valid_str}]")
    print(f"  Iterations: {iters_enh}, Columns: {cols_enh}, Time: {time_enh:.1f}s")
    print(f"  API calls: {timer_enh['num_api_calls']}, "
          f"Cols found: {timer_enh['total_columns_found']}, "
          f"Avg cols/call: {timer_enh['avg_columns_per_call']:.1f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Method':<30} {'chi':<6} {'Valid':<8} {'Iters':<8} {'Cols':<8} {'Time(s)':<10}")
    print("-"*70)
    print(f"{'Greedy':<30} {greedy_chi:<6} {'Y':<8} {'-':<8} {'-':<8} {greedy_time:<10.3f}")
    v1 = 'Y' if valid_base else 'N'
    chi1 = chi_base if chi_base else '-'
    print(f"{'Dirac CG (baseline)':<30} {chi1:<6} {v1:<8} {iters_base:<8} {cols_base:<8} {time_base:<10.1f}")
    v2 = 'Y' if valid_enh else 'N'
    chi2 = chi_enh if chi_enh else '-'
    print(f"{'Dirac CG (enhanced)':<30} {chi2:<6} {v2:<8} {iters_enh:<8} {cols_enh:<8} {time_enh:<10.1f}")

    print("\n" + "-"*70)
    print("Extraction improvement:")
    if timer_base['num_api_calls'] > 0 and timer_enh['num_api_calls'] > 0:
        base_cols_per_call = timer_base['avg_columns_per_call']
        enh_cols_per_call = timer_enh['avg_columns_per_call']
        improvement = enh_cols_per_call / base_cols_per_call if base_cols_per_call > 0 else 0
        print(f"  Baseline: {base_cols_per_call:.1f} cols/call")
        print(f"  Enhanced: {enh_cols_per_call:.1f} cols/call ({improvement:.1f}x improvement)")

    print("\nChi comparison:")
    if chi_base and chi_enh:
        if chi_enh < chi_base:
            print(f"  Enhanced is BETTER: {chi_enh} < {chi_base} (baseline)")
        elif chi_enh == chi_base:
            print(f"  Enhanced EQUALS baseline: {chi_enh}")
        else:
            print(f"  Enhanced is WORSE: {chi_enh} > {chi_base} (baseline)")

        if chi_enh < greedy_chi:
            print(f"  Enhanced BEATS greedy: {chi_enh} < {greedy_chi}")
        elif chi_enh == greedy_chi:
            print(f"  Enhanced MATCHES greedy: {chi_enh}")
        else:
            print(f"  Enhanced LOSES to greedy: {chi_enh} > {greedy_chi}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
