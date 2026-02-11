#!/usr/bin/env python
"""Experiments: Bomze regularization + CG acceleration.

Stages:
  S1 — Regularization sweep (Dirac, ~10 API calls)
  S2 — Full CG with regularization (Dirac, ~60 API calls)
  S3 — CG acceleration with classical oracle (free)
  S4 — Combined best (Dirac, ~40 API calls)

Usage:
    # Classical-only acceleration test (free, start here)
    uv run python scripts/experiment_bomze_cg_accel.py --stage S3 --er-sizes 30 40 50

    # Regularization sweep (10 Dirac calls)
    source ~/.zshrc
    uv run python scripts/experiment_bomze_cg_accel.py --stage S1 --er-sizes 30 40

    # Full CG with reg (Dirac)
    uv run python scripts/experiment_bomze_cg_accel.py --stage S2 --er-sizes 30 40

    # Combined best
    uv run python scripts/experiment_bomze_cg_accel.py --stage S4 --er-sizes 30 40

    # Save results
    uv run python scripts/experiment_bomze_cg_accel.py --stage S3 --json results/bomze_s3.json
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from quantum_colgen.graphs import erdos_renyi
from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.master_problem import solve_rmp
from quantum_colgen.pricing.classical_lp import ClassicalLPPricingOracle
from quantum_colgen.timing import OracleTimer

# Optional Dirac imports
try:
    from quantum_colgen.pricing.dirac_oracle import (
        DiracPricingOracle,
        _construct_gibbons_matrix,
        _local_search,
        _greedy_prune_dual_desc,
        DIRAC_AVAILABLE,
    )
    from eqc_models.solvers import Dirac3ContinuousCloudSolver
    from eqc_models.base import QuadraticModel
except ImportError:
    DIRAC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class S1Result:
    """Single-call regularization sweep result."""
    graph_name: str
    c_value: float
    unique_is: int
    profitable_is: int
    avg_is_size: float
    jaccard_diversity: float
    wall_seconds: float


@dataclass
class CGResult:
    """Full CG run result."""
    graph_name: str
    config: str
    chi: Optional[int]
    valid: bool
    iterations: int
    columns_generated: int
    oracle_calls_skipped: int
    wall_seconds: float
    rmp_obj_trace: List[float] = field(default_factory=list)


@dataclass
class ExperimentResults:
    """All experiment results."""
    stage: str
    s1_results: List[S1Result] = field(default_factory=list)
    cg_results: List[CGResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(n: int, p: float, seed: int = 42) -> nx.Graph:
    return erdos_renyi(n, p, seed=seed)


def _greedy_chi(graph: nx.Graph) -> int:
    coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
    return max(coloring.values()) + 1 if coloring else 0


def _jaccard_diversity(sets: List[set]) -> float:
    """Average pairwise Jaccard distance across a list of sets."""
    if len(sets) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = len(sets[i] | sets[j])
            if union > 0:
                total += 1.0 - len(sets[i] & sets[j]) / union
            count += 1
    return total / count if count else 0.0


# ---------------------------------------------------------------------------
# S1: Regularization sweep
# ---------------------------------------------------------------------------

def run_s1(
    er_sizes: List[int],
    edge_prob: float = 0.3,
    seed: int = 42,
    c_values: Optional[List[float]] = None,
    num_samples: int = 100,
    verbose: bool = False,
) -> List[S1Result]:
    """Regularization sweep: one Dirac call per (graph, c)."""
    if not DIRAC_AVAILABLE:
        print("ERROR: Dirac not available. Run `source ~/.zshrc` first.")
        return []

    if c_values is None:
        c_values = [0.0, 0.1, 0.2, 0.3, 0.5]

    results: List[S1Result] = []

    for n in er_sizes:
        graph = _make_graph(n, edge_prob, seed=seed)
        graph_name = f"ER({n},{edge_prob})"
        print(f"\n{'='*60}")
        print(f"S1: {graph_name}  n={n} m={graph.number_of_edges()}")
        print(f"{'='*60}")

        # Get initial duals from singleton RMP
        singleton_cols = [frozenset([i]) for i in range(n)]
        _, dual_vars = solve_rmp(singleton_cols, n)
        if dual_vars is None:
            print("  Failed to get initial duals, skipping")
            continue

        for c in c_values:
            oracle = DiracPricingOracle(
                method="gibbons",
                num_samples=num_samples,
                multi_prune=True,
                randomized_rounding=True,
                num_random_rounds=10,
                random_seed=42,
                regularization_c=c,
            )

            t0 = time.monotonic()
            columns = oracle.solve(graph, dual_vars)
            elapsed = time.monotonic() - t0

            profitable = [
                col for col in columns
                if sum(dual_vars[v] for v in col) > 1 + 1e-5
            ]
            sizes = [len(col) for col in columns] if columns else [0]
            diversity = _jaccard_diversity([set(col) for col in columns]) if columns else 0.0

            result = S1Result(
                graph_name=graph_name,
                c_value=c,
                unique_is=len(columns),
                profitable_is=len(profitable),
                avg_is_size=sum(sizes) / len(sizes) if sizes else 0,
                jaccard_diversity=round(diversity, 3),
                wall_seconds=round(elapsed, 2),
            )
            results.append(result)

            print(f"  c={c:.1f}: {len(columns)} unique IS, "
                  f"{len(profitable)} profitable, "
                  f"avg_size={result.avg_is_size:.1f}, "
                  f"diversity={result.jaccard_diversity:.3f}, "
                  f"{elapsed:.1f}s")

    return results


# ---------------------------------------------------------------------------
# S2: Full CG with regularization
# ---------------------------------------------------------------------------

def run_s2(
    er_sizes: List[int],
    edge_prob: float = 0.3,
    seed: int = 42,
    c_values: Optional[List[float]] = None,
    max_iterations: int = 500,
    verbose: bool = False,
) -> List[CGResult]:
    """Full CG with different regularization values."""
    if not DIRAC_AVAILABLE:
        print("ERROR: Dirac not available. Run `source ~/.zshrc` first.")
        return []

    if c_values is None:
        c_values = [0.0, 0.1, 0.2, 0.3]

    results: List[CGResult] = []

    for n in er_sizes:
        graph = _make_graph(n, edge_prob, seed=seed)
        graph_name = f"ER({n},{edge_prob})"
        greedy = _greedy_chi(graph)
        print(f"\n{'='*60}")
        print(f"S2: {graph_name}  n={n} m={graph.number_of_edges()}  greedy={greedy}")
        print(f"{'='*60}")

        for c in c_values:
            config = f"c={c}"
            print(f"\n  Config: {config}")

            oracle = DiracPricingOracle(
                method="gibbons",
                num_samples=100,
                multi_prune=True,
                randomized_rounding=True,
                random_seed=42,
                regularization_c=c,
            )

            t0 = time.monotonic()
            chi, coloring, stats = column_generation(
                graph, oracle,
                max_iterations=max_iterations,
                verbose=verbose,
            )
            elapsed = time.monotonic() - t0

            valid = verify_coloring(graph, coloring) if coloring else False

            result = CGResult(
                graph_name=graph_name,
                config=config,
                chi=chi,
                valid=valid,
                iterations=stats.get("iterations", 0),
                columns_generated=stats.get("columns_generated", 0),
                oracle_calls_skipped=stats.get("oracle_calls_skipped", 0),
                wall_seconds=round(elapsed, 2),
                rmp_obj_trace=stats.get("rmp_obj_trace", []),
            )
            results.append(result)

            valid_str = "VALID" if valid else "INVALID"
            print(f"    chi={chi} iters={result.iterations} "
                  f"cols={result.columns_generated} "
                  f"time={elapsed:.1f}s [{valid_str}]")

    return results


# ---------------------------------------------------------------------------
# S3: CG acceleration (classical, free)
# ---------------------------------------------------------------------------

def run_s3(
    er_sizes: List[int],
    edge_prob: float = 0.3,
    seed: int = 42,
    max_iterations: int = 500,
    verbose: bool = False,
) -> List[CGResult]:
    """CG acceleration experiments with classical LP oracle (free)."""
    results: List[CGResult] = []

    alphas = [None, 0.1, 0.3, 0.5, 0.7]
    aging_thresholds = [None, 0.01, 0.05, 0.1]

    for n in er_sizes:
        graph = _make_graph(n, edge_prob, seed=seed)
        graph_name = f"ER({n},{edge_prob})"
        greedy = _greedy_chi(graph)
        print(f"\n{'='*60}")
        print(f"S3: {graph_name}  n={n} m={graph.number_of_edges()}  greedy={greedy}")
        print(f"{'='*60}")

        # --- Dual smoothing sweep ---
        print("\n  Dual smoothing sweep:")
        for alpha in alphas:
            config = f"alpha={alpha}" if alpha is not None else "baseline"
            oracle = ClassicalLPPricingOracle(random_seed=42)

            t0 = time.monotonic()
            chi, coloring, stats = column_generation(
                graph, oracle,
                max_iterations=max_iterations,
                dual_smoothing_alpha=alpha,
                verbose=verbose,
            )
            elapsed = time.monotonic() - t0

            valid = verify_coloring(graph, coloring) if coloring else False

            result = CGResult(
                graph_name=graph_name,
                config=config,
                chi=chi,
                valid=valid,
                iterations=stats.get("iterations", 0),
                columns_generated=stats.get("columns_generated", 0),
                oracle_calls_skipped=stats.get("oracle_calls_skipped", 0),
                wall_seconds=round(elapsed, 3),
                rmp_obj_trace=stats.get("rmp_obj_trace", []),
            )
            results.append(result)

            valid_str = "VALID" if valid else "INVALID"
            print(f"    {config:<15} chi={chi} iters={result.iterations:>3} "
                  f"cols={result.columns_generated:>4} "
                  f"time={elapsed:.3f}s [{valid_str}]")

        # --- Subproblem aging sweep ---
        print("\n  Subproblem aging sweep:")
        for threshold in aging_thresholds:
            config = f"aging={threshold}" if threshold is not None else "baseline"
            oracle = ClassicalLPPricingOracle(random_seed=42)

            t0 = time.monotonic()
            chi, coloring, stats = column_generation(
                graph, oracle,
                max_iterations=max_iterations,
                subproblem_aging_threshold=threshold,
                verbose=verbose,
            )
            elapsed = time.monotonic() - t0

            valid = verify_coloring(graph, coloring) if coloring else False

            result = CGResult(
                graph_name=graph_name,
                config=config,
                chi=chi,
                valid=valid,
                iterations=stats.get("iterations", 0),
                columns_generated=stats.get("columns_generated", 0),
                oracle_calls_skipped=stats.get("oracle_calls_skipped", 0),
                wall_seconds=round(elapsed, 3),
                rmp_obj_trace=stats.get("rmp_obj_trace", []),
            )
            results.append(result)

            valid_str = "VALID" if valid else "INVALID"
            skipped = stats.get("oracle_calls_skipped", 0)
            print(f"    {config:<15} chi={chi} iters={result.iterations:>3} "
                  f"cols={result.columns_generated:>4} skipped={skipped:>2} "
                  f"time={elapsed:.3f}s [{valid_str}]")

        # --- Combined: best alpha + best threshold ---
        print("\n  Combined (alpha=0.3 + aging=0.05):")
        oracle = ClassicalLPPricingOracle(random_seed=42)

        t0 = time.monotonic()
        chi, coloring, stats = column_generation(
            graph, oracle,
            max_iterations=max_iterations,
            dual_smoothing_alpha=0.3,
            subproblem_aging_threshold=0.05,
            verbose=verbose,
        )
        elapsed = time.monotonic() - t0

        valid = verify_coloring(graph, coloring) if coloring else False

        result = CGResult(
            graph_name=graph_name,
            config="alpha=0.3+aging=0.05",
            chi=chi,
            valid=valid,
            iterations=stats.get("iterations", 0),
            columns_generated=stats.get("columns_generated", 0),
            oracle_calls_skipped=stats.get("oracle_calls_skipped", 0),
            wall_seconds=round(elapsed, 3),
            rmp_obj_trace=stats.get("rmp_obj_trace", []),
        )
        results.append(result)

        valid_str = "VALID" if valid else "INVALID"
        skipped = stats.get("oracle_calls_skipped", 0)
        print(f"    combined        chi={chi} iters={result.iterations:>3} "
              f"cols={result.columns_generated:>4} skipped={skipped:>2} "
              f"time={elapsed:.3f}s [{valid_str}]")

    return results


# ---------------------------------------------------------------------------
# S4: Combined best (Dirac)
# ---------------------------------------------------------------------------

def run_s4(
    er_sizes: List[int],
    edge_prob: float = 0.3,
    seed: int = 42,
    best_c: float = 0.1,
    best_alpha: Optional[float] = 0.3,
    best_aging: Optional[float] = 0.05,
    max_iterations: int = 500,
    verbose: bool = False,
) -> List[CGResult]:
    """Combined best: regularization + CG acceleration with Dirac."""
    if not DIRAC_AVAILABLE:
        print("ERROR: Dirac not available. Run `source ~/.zshrc` first.")
        return []

    results: List[CGResult] = []

    configs = [
        ("baseline", 0.0, None, None),
        (f"reg_c={best_c}", best_c, None, None),
        (f"accel_a={best_alpha}_t={best_aging}", 0.0, best_alpha, best_aging),
        (f"combined_c={best_c}_a={best_alpha}_t={best_aging}", best_c, best_alpha, best_aging),
    ]

    for n in er_sizes:
        graph = _make_graph(n, edge_prob, seed=seed)
        graph_name = f"ER({n},{edge_prob})"
        greedy = _greedy_chi(graph)
        print(f"\n{'='*60}")
        print(f"S4: {graph_name}  n={n} m={graph.number_of_edges()}  greedy={greedy}")
        print(f"{'='*60}")

        for config_name, c, alpha, aging in configs:
            print(f"\n  Config: {config_name}")

            oracle = DiracPricingOracle(
                method="gibbons",
                num_samples=100,
                multi_prune=True,
                randomized_rounding=True,
                random_seed=42,
                regularization_c=c,
            )

            t0 = time.monotonic()
            chi, coloring, stats = column_generation(
                graph, oracle,
                max_iterations=max_iterations,
                dual_smoothing_alpha=alpha,
                subproblem_aging_threshold=aging,
                verbose=verbose,
            )
            elapsed = time.monotonic() - t0

            valid = verify_coloring(graph, coloring) if coloring else False

            result = CGResult(
                graph_name=graph_name,
                config=config_name,
                chi=chi,
                valid=valid,
                iterations=stats.get("iterations", 0),
                columns_generated=stats.get("columns_generated", 0),
                oracle_calls_skipped=stats.get("oracle_calls_skipped", 0),
                wall_seconds=round(elapsed, 2),
                rmp_obj_trace=stats.get("rmp_obj_trace", []),
            )
            results.append(result)

            valid_str = "VALID" if valid else "INVALID"
            skipped = stats.get("oracle_calls_skipped", 0)
            print(f"    chi={chi} iters={result.iterations} "
                  f"cols={result.columns_generated} skipped={skipped} "
                  f"time={elapsed:.1f}s [{valid_str}]")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_s1_summary(results: List[S1Result]) -> None:
    print("\n" + "=" * 75)
    print("S1 SUMMARY: Regularization Sweep")
    print("=" * 75)
    print(f"{'Graph':<15} {'c':>5} {'unique_IS':>10} {'profit':>8} "
          f"{'avg_size':>9} {'diversity':>10} {'time':>6}")
    print("-" * 75)
    for r in results:
        print(f"{r.graph_name:<15} {r.c_value:>5.1f} {r.unique_is:>10} "
              f"{r.profitable_is:>8} {r.avg_is_size:>9.1f} "
              f"{r.jaccard_diversity:>10.3f} {r.wall_seconds:>6.1f}")


def print_cg_summary(results: List[CGResult], title: str) -> None:
    print(f"\n{'=' * 85}")
    print(f"{title}")
    print("=" * 85)
    print(f"{'Graph':<15} {'Config':<25} {'chi':>4} {'Valid':>6} {'Iters':>6} "
          f"{'Cols':>5} {'Skip':>5} {'Time':>7}")
    print("-" * 85)
    for r in results:
        valid_str = "Y" if r.valid else "N"
        chi_str = str(r.chi) if r.chi is not None else "-"
        print(f"{r.graph_name:<15} {r.config:<25} {chi_str:>4} {valid_str:>6} "
              f"{r.iterations:>6} {r.columns_generated:>5} "
              f"{r.oracle_calls_skipped:>5} {r.wall_seconds:>7.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bomze regularization + CG acceleration experiments"
    )
    parser.add_argument(
        "--stage", type=str, choices=["S1", "S2", "S3", "S4"],
        help="Run a specific stage"
    )
    parser.add_argument("--all", action="store_true", help="Run all stages")
    parser.add_argument("--er-sizes", nargs="+", type=int, default=[30, 40])
    parser.add_argument("--edge-prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", type=str, help="Save results to JSON")

    args = parser.parse_args()

    if not args.stage and not args.all:
        parser.error("Specify --stage S1/S2/S3/S4 or --all")

    all_results = ExperimentResults(stage=args.stage or "all")

    stages = [args.stage] if args.stage else ["S3", "S1", "S2", "S4"]

    for stage in stages:
        if stage == "S1":
            s1 = run_s1(
                args.er_sizes, args.edge_prob, args.seed,
                verbose=args.verbose,
            )
            all_results.s1_results.extend(s1)
            print_s1_summary(s1)

        elif stage == "S2":
            s2 = run_s2(
                args.er_sizes, args.edge_prob, args.seed,
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
            all_results.cg_results.extend(s2)
            print_cg_summary(s2, "S2 SUMMARY: Full CG with Regularization")

        elif stage == "S3":
            s3 = run_s3(
                args.er_sizes, args.edge_prob, args.seed,
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
            all_results.cg_results.extend(s3)
            print_cg_summary(s3, "S3 SUMMARY: CG Acceleration (Classical)")

        elif stage == "S4":
            s4 = run_s4(
                args.er_sizes, args.edge_prob, args.seed,
                max_iterations=args.max_iterations,
                verbose=args.verbose,
            )
            all_results.cg_results.extend(s4)
            print_cg_summary(s4, "S4 SUMMARY: Combined Best (Dirac)")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(asdict(all_results), f, indent=2)
        print(f"\nResults saved to {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
