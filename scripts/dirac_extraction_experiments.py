#!/usr/bin/env python
"""Experiments to improve Dirac IS extraction from Motzkin-Straus solutions.

This script tests multiple extraction strategies to maximize the number of
useful independent sets (columns) extracted from Dirac QP solutions.

Strategies tested:
1. Baseline (current): 5 thresholds, greedy prune by dual weight desc
2. Fine-grained thresholds: 15+ thresholds
3. Randomized rounding: probabilistic inclusion based on solution values
4. Different pruning orders: ascending, random, size-based
5. Sub-IS extraction: enumerate profitable subsets of each IS
6. Relaxed profitability: accept columns with sum(duals) > threshold < 1.0

Usage:
    # Analyze a single graph with live Dirac API
    source ~/.zshrc
    uv run python scripts/dirac_extraction_experiments.py --nodes 15 --edge-prob 0.3

    # Run all experiments
    uv run python scripts/dirac_extraction_experiments.py --all --nodes 30 --edge-prob 0.3

    # Save detailed analysis
    uv run python scripts/dirac_extraction_experiments.py --all --nodes 30 --save results/extraction_analysis.json
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from quantum_colgen.graphs import erdos_renyi
from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.master_problem import solve_final_ilp
from quantum_colgen.pricing.base import PricingOracle
from quantum_colgen.timing import OracleTimer


# Try to import Dirac dependencies
try:
    from quantum_colgen.pricing.dirac_oracle import (
        DiracPricingOracle,
        _local_search,
        _construct_gibbons_matrix,
        DIRAC_AVAILABLE,
    )
    from eqc_models.solvers import Dirac3ContinuousCloudSolver
    from eqc_models.base import QuadraticModel
except ImportError:
    DIRAC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Threshold configurations
# ---------------------------------------------------------------------------

BASELINE_THRESHOLDS = [0.005, 0.01, 0.05, 0.1, 0.2]

FINE_THRESHOLDS = [
    0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07,
    0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5
]

ULTRA_FINE_THRESHOLDS = [
    0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.04,
    0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.25, 0.3, 0.4, 0.5
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractionStats:
    """Statistics for a single extraction strategy."""
    strategy_name: str
    unique_is_count: int  # before profitability filter
    profitable_count: int  # after profitability filter (sum > 1)
    relaxed_count: int  # sum > 0.5
    very_relaxed_count: int  # sum > 0.0
    max_is_size: int
    avg_is_size: float
    extraction_time_ms: float
    # Per-threshold or per-variant breakdown
    breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class ColumnGenerationStats:
    """Statistics from running full CG with a strategy."""
    strategy_name: str
    chi: Optional[int]
    valid: bool
    iterations: int
    columns_generated: int
    wall_seconds: float


@dataclass
class ExperimentResult:
    """Full experiment result for a graph."""
    graph_name: str
    n_nodes: int
    n_edges: int
    greedy_chi: int
    num_dirac_samples: int
    extraction_stats: List[ExtractionStats] = field(default_factory=list)
    cg_stats: List[ColumnGenerationStats] = field(default_factory=list)
    raw_solution_analysis: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extraction strategies
# ---------------------------------------------------------------------------

def greedy_prune_by_dual_desc(
    support: Set[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
) -> Set[int]:
    """Greedy prune: highest dual weight first (current baseline)."""
    pruned: Set[int] = set()
    for v in sorted(support, key=lambda v: dual_vars[v], reverse=True):
        if not any(graph.has_edge(v, u) for u in pruned):
            pruned.add(v)
    return pruned


def greedy_prune_by_dual_asc(
    support: Set[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
) -> Set[int]:
    """Greedy prune: lowest dual weight first (gives different IS)."""
    pruned: Set[int] = set()
    for v in sorted(support, key=lambda v: dual_vars[v], reverse=False):
        if not any(graph.has_edge(v, u) for u in pruned):
            pruned.add(v)
    return pruned


def greedy_prune_random(
    support: Set[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
    rng: random.Random,
) -> Set[int]:
    """Greedy prune: random order."""
    nodes = list(support)
    rng.shuffle(nodes)
    pruned: Set[int] = set()
    for v in nodes:
        if not any(graph.has_edge(v, u) for u in pruned):
            pruned.add(v)
    return pruned


def greedy_prune_by_degree_desc(
    support: Set[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
) -> Set[int]:
    """Greedy prune: highest degree (in subgraph) first."""
    subgraph = graph.subgraph(support)
    pruned: Set[int] = set()
    for v in sorted(support, key=lambda v: subgraph.degree(v), reverse=True):
        if not any(graph.has_edge(v, u) for u in pruned):
            pruned.add(v)
    return pruned


def extract_baseline(
    solutions: List[np.ndarray],
    node_list: List[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
    thresholds: List[float] = BASELINE_THRESHOLDS,
    local_search_passes: int = 5,
) -> Tuple[List[Set[int]], Dict[str, int]]:
    """Baseline extraction: current implementation."""
    seen: set = set()
    all_is: List[Set[int]] = []
    breakdown: Dict[str, int] = defaultdict(int)

    for x in solutions:
        for threshold in thresholds:
            support = {node_list[i] for i in range(len(x)) if x[i] > threshold}
            if not support:
                continue

            pruned = greedy_prune_by_dual_desc(support, graph, dual_vars)
            pruned = _local_search(graph, pruned, dual_vars, max_passes=local_search_passes)

            sig = frozenset(pruned)
            if sig not in seen:
                seen.add(sig)
                all_is.append(set(pruned))
                breakdown[f"t={threshold}"] += 1

    return all_is, dict(breakdown)


def extract_fine_thresholds(
    solutions: List[np.ndarray],
    node_list: List[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
    local_search_passes: int = 5,
) -> Tuple[List[Set[int]], Dict[str, int]]:
    """Fine-grained thresholds extraction."""
    return extract_baseline(
        solutions, node_list, graph, dual_vars,
        thresholds=FINE_THRESHOLDS,
        local_search_passes=local_search_passes,
    )


def extract_ultra_fine_thresholds(
    solutions: List[np.ndarray],
    node_list: List[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
    local_search_passes: int = 5,
) -> Tuple[List[Set[int]], Dict[str, int]]:
    """Ultra-fine-grained thresholds extraction."""
    return extract_baseline(
        solutions, node_list, graph, dual_vars,
        thresholds=ULTRA_FINE_THRESHOLDS,
        local_search_passes=local_search_passes,
    )


def extract_multi_prune_orders(
    solutions: List[np.ndarray],
    node_list: List[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
    local_search_passes: int = 5,
    num_random_trials: int = 5,
    seed: int = 42,
) -> Tuple[List[Set[int]], Dict[str, int]]:
    """Try multiple pruning strategies per (solution, threshold) pair."""
    rng = random.Random(seed)
    seen: set = set()
    all_is: List[Set[int]] = []
    breakdown: Dict[str, int] = defaultdict(int)

    prune_funcs = [
        ("dual_desc", lambda s, g, d: greedy_prune_by_dual_desc(s, g, d)),
        ("dual_asc", lambda s, g, d: greedy_prune_by_dual_asc(s, g, d)),
        ("degree_desc", lambda s, g, d: greedy_prune_by_degree_desc(s, g, d)),
    ]
    # Add random trials
    for i in range(num_random_trials):
        trial_rng = random.Random(seed + i)
        prune_funcs.append(
            (f"random_{i}", lambda s, g, d, r=trial_rng: greedy_prune_random(s, g, d, r))
        )

    for x in solutions:
        for threshold in FINE_THRESHOLDS:
            support = {node_list[i] for i in range(len(x)) if x[i] > threshold}
            if not support:
                continue

            for prune_name, prune_func in prune_funcs:
                pruned = prune_func(support, graph, dual_vars)
                pruned = _local_search(graph, pruned, dual_vars, max_passes=local_search_passes)

                sig = frozenset(pruned)
                if sig not in seen:
                    seen.add(sig)
                    all_is.append(set(pruned))
                    breakdown[prune_name] += 1

    return all_is, dict(breakdown)


def extract_randomized_rounding(
    solutions: List[np.ndarray],
    node_list: List[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
    local_search_passes: int = 5,
    num_rounds: int = 20,
    seed: int = 42,
) -> Tuple[List[Set[int]], Dict[str, int]]:
    """Probabilistic rounding: include node i with probability ~ x[i]."""
    rng = random.Random(seed)
    seen: set = set()
    all_is: List[Set[int]] = []
    breakdown: Dict[str, int] = defaultdict(int)

    for sol_idx, x in enumerate(solutions):
        x_norm = x / (x.sum() + 1e-10)  # normalize to probabilities

        for round_idx in range(num_rounds):
            # Sample nodes proportionally to solution values
            support = set()
            for i, p in enumerate(x_norm):
                if rng.random() < p * 3:  # scale up probability
                    support.add(node_list[i])

            if not support:
                continue

            pruned = greedy_prune_by_dual_desc(support, graph, dual_vars)
            pruned = _local_search(graph, pruned, dual_vars, max_passes=local_search_passes)

            sig = frozenset(pruned)
            if sig not in seen:
                seen.add(sig)
                all_is.append(set(pruned))
                breakdown[f"sol_{sol_idx}"] += 1

    return all_is, dict(breakdown)


def extract_sub_is(
    independent_sets: List[Set[int]],
    graph: nx.Graph,
    dual_vars: np.ndarray,
    min_profit: float = 0.5,
    max_removals: int = 3,
) -> Tuple[List[Set[int]], Dict[str, int]]:
    """Extract profitable subsets from each IS.

    For each IS S, try removing 1, 2, or 3 nodes and check if result is still profitable.
    """
    seen: set = set()
    sub_is: List[Set[int]] = []
    breakdown: Dict[str, int] = defaultdict(int)

    for is_set in independent_sets:
        sig = frozenset(is_set)
        if sig not in seen:
            seen.add(sig)
            sub_is.append(is_set)
            breakdown["original"] += 1

        # Try removing k nodes
        for k in range(1, min(max_removals + 1, len(is_set))):
            for to_remove in combinations(is_set, k):
                subset = is_set - set(to_remove)
                if not subset:
                    continue

                sig = frozenset(subset)
                if sig not in seen:
                    profit = sum(dual_vars[v] for v in subset)
                    if profit > min_profit:
                        seen.add(sig)
                        sub_is.append(subset)
                        breakdown[f"sub_{k}"] += 1

    return sub_is, dict(breakdown)


def extract_combined_best(
    solutions: List[np.ndarray],
    node_list: List[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
    local_search_passes: int = 5,
    seed: int = 42,
) -> Tuple[List[Set[int]], Dict[str, int]]:
    """Combined strategy: fine thresholds + multi-prune + randomized rounding."""
    seen: set = set()
    all_is: List[Set[int]] = []
    breakdown: Dict[str, int] = defaultdict(int)

    # 1. Fine thresholds with multi-prune
    multi_is, multi_bd = extract_multi_prune_orders(
        solutions, node_list, graph, dual_vars,
        local_search_passes=local_search_passes,
        num_random_trials=3,
        seed=seed,
    )
    for is_set in multi_is:
        sig = frozenset(is_set)
        if sig not in seen:
            seen.add(sig)
            all_is.append(is_set)
            breakdown["multi_prune"] += 1

    # 2. Randomized rounding
    rand_is, rand_bd = extract_randomized_rounding(
        solutions, node_list, graph, dual_vars,
        local_search_passes=local_search_passes,
        num_rounds=10,
        seed=seed + 100,
    )
    for is_set in rand_is:
        sig = frozenset(is_set)
        if sig not in seen:
            seen.add(sig)
            all_is.append(is_set)
            breakdown["random_round"] += 1

    return all_is, dict(breakdown)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def analyze_solutions(solutions: List[np.ndarray]) -> Dict[str, Any]:
    """Analyze raw Dirac solution vectors."""
    if not solutions:
        return {}

    all_values = np.concatenate(solutions)
    nonzero_values = all_values[all_values > 1e-10]

    # Value distribution
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    value_percentiles = {
        f"p{p}": float(np.percentile(nonzero_values, p))
        for p in percentiles
    } if len(nonzero_values) > 0 else {}

    # Per-solution stats
    per_solution_stats = []
    for i, x in enumerate(solutions[:10]):  # First 10 only
        nonzero = x[x > 1e-10]
        per_solution_stats.append({
            "idx": i,
            "n_nonzero": len(nonzero),
            "max": float(x.max()),
            "mean_nonzero": float(nonzero.mean()) if len(nonzero) > 0 else 0,
        })

    return {
        "num_samples": len(solutions),
        "solution_dim": len(solutions[0]) if solutions else 0,
        "total_nonzero_entries": len(nonzero_values),
        "avg_nonzero_per_sample": len(nonzero_values) / len(solutions) if solutions else 0,
        "value_percentiles": value_percentiles,
        "first_10_samples": per_solution_stats,
    }


def compute_extraction_stats(
    all_is: List[Set[int]],
    dual_vars: np.ndarray,
    breakdown: Dict[str, int],
    strategy_name: str,
    extraction_time_ms: float,
) -> ExtractionStats:
    """Compute stats for extracted independent sets."""
    if not all_is:
        return ExtractionStats(
            strategy_name=strategy_name,
            unique_is_count=0,
            profitable_count=0,
            relaxed_count=0,
            very_relaxed_count=0,
            max_is_size=0,
            avg_is_size=0.0,
            extraction_time_ms=extraction_time_ms,
            breakdown=breakdown,
        )

    profits = [sum(dual_vars[v] for v in is_set) for is_set in all_is]
    sizes = [len(is_set) for is_set in all_is]

    return ExtractionStats(
        strategy_name=strategy_name,
        unique_is_count=len(all_is),
        profitable_count=sum(1 for p in profits if p > 1 + 1e-5),
        relaxed_count=sum(1 for p in profits if p > 0.5),
        very_relaxed_count=sum(1 for p in profits if p > 0),
        max_is_size=max(sizes),
        avg_is_size=sum(sizes) / len(sizes),
        extraction_time_ms=extraction_time_ms,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Experimental oracle wrapper
# ---------------------------------------------------------------------------

class ExperimentalDiracOracle(PricingOracle):
    """Dirac oracle with configurable extraction strategy for experiments."""

    def __init__(
        self,
        extraction_strategy: str = "baseline",
        profitability_threshold: float = 1.0,
        num_samples: int = 100,
        relax_schedule: int = 2,
        local_search_passes: int = 5,
        seed: int = 42,
    ):
        if not DIRAC_AVAILABLE:
            raise ImportError("Dirac not available")

        self.extraction_strategy = extraction_strategy
        self.profitability_threshold = profitability_threshold
        self.num_samples = num_samples
        self.relax_schedule = relax_schedule
        self.local_search_passes = local_search_passes
        self.seed = seed
        self.timer = OracleTimer()

        # Cache last Dirac response for analysis
        self.last_solutions: Optional[List[np.ndarray]] = None
        self.last_node_list: Optional[List[int]] = None

    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        pos_nodes = [v for v in graph.nodes() if dual_vars[v] > 1e-5]
        if not pos_nodes:
            return []

        subgraph = graph.subgraph(pos_nodes)
        if not subgraph.edges():
            total = sum(dual_vars[v] for v in pos_nodes)
            return [set(pos_nodes)] if total > self.profitability_threshold + 1e-5 else []

        # Build Gibbons matrix and solve
        weights = {v: float(dual_vars[v]) for v in pos_nodes}
        complement = nx.complement(subgraph)
        B = _construct_gibbons_matrix(complement, weights)

        n = B.shape[0]
        C = np.zeros(n, dtype=np.float64)
        J = B.astype(np.float64)

        model = QuadraticModel(C, J)
        model.upper_bound = np.ones(n, dtype=np.float64)
        solver = Dirac3ContinuousCloudSolver()

        t0 = time.monotonic()
        response = solver.solve(
            model,
            sum_constraint=1,
            num_samples=self.num_samples,
            relaxation_schedule=self.relax_schedule,
        )
        api_seconds = time.monotonic() - t0

        if not (response and "results" in response and "solutions" in response["results"]):
            self.timer.record(api_seconds=api_seconds)
            return []

        raw_solutions = response["results"]["solutions"]
        if not raw_solutions:
            self.timer.record(api_seconds=api_seconds)
            return []

        solutions = [np.array(s, dtype=np.float64) for s in raw_solutions]
        node_list = sorted(subgraph.nodes())

        # Cache for analysis
        self.last_solutions = solutions
        self.last_node_list = node_list

        # Extract using configured strategy
        t0 = time.monotonic()
        all_is, _ = self._extract(solutions, node_list, graph, dual_vars)
        extract_seconds = time.monotonic() - t0

        # Filter by profitability threshold
        profitable = []
        for is_set in all_is:
            profit = sum(dual_vars[v] for v in is_set)
            if profit > self.profitability_threshold + 1e-5:
                profitable.append(is_set)

        self.timer.record(
            api_seconds=api_seconds,
            extract_seconds=extract_seconds,
            columns_found=len(profitable),
        )
        return profitable

    def _extract(
        self,
        solutions: List[np.ndarray],
        node_list: List[int],
        graph: nx.Graph,
        dual_vars: np.ndarray,
    ) -> Tuple[List[Set[int]], Dict[str, int]]:
        """Extract IS using configured strategy."""
        if self.extraction_strategy == "baseline":
            return extract_baseline(
                solutions, node_list, graph, dual_vars,
                local_search_passes=self.local_search_passes,
            )
        elif self.extraction_strategy == "fine_thresholds":
            return extract_fine_thresholds(
                solutions, node_list, graph, dual_vars,
                local_search_passes=self.local_search_passes,
            )
        elif self.extraction_strategy == "ultra_fine":
            return extract_ultra_fine_thresholds(
                solutions, node_list, graph, dual_vars,
                local_search_passes=self.local_search_passes,
            )
        elif self.extraction_strategy == "multi_prune":
            return extract_multi_prune_orders(
                solutions, node_list, graph, dual_vars,
                local_search_passes=self.local_search_passes,
                seed=self.seed,
            )
        elif self.extraction_strategy == "randomized":
            return extract_randomized_rounding(
                solutions, node_list, graph, dual_vars,
                local_search_passes=self.local_search_passes,
                seed=self.seed,
            )
        elif self.extraction_strategy == "combined":
            return extract_combined_best(
                solutions, node_list, graph, dual_vars,
                local_search_passes=self.local_search_passes,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown strategy: {self.extraction_strategy}")


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_extraction_analysis(
    graph: nx.Graph,
    dual_vars: np.ndarray,
    solutions: List[np.ndarray],
    node_list: List[int],
    verbose: bool = False,
) -> List[ExtractionStats]:
    """Run all extraction strategies and compare results."""
    stats_list = []

    strategies = [
        ("baseline", lambda: extract_baseline(
            solutions, node_list, graph, dual_vars)),
        ("fine_thresholds", lambda: extract_fine_thresholds(
            solutions, node_list, graph, dual_vars)),
        ("ultra_fine", lambda: extract_ultra_fine_thresholds(
            solutions, node_list, graph, dual_vars)),
        ("multi_prune", lambda: extract_multi_prune_orders(
            solutions, node_list, graph, dual_vars)),
        ("randomized", lambda: extract_randomized_rounding(
            solutions, node_list, graph, dual_vars)),
        ("combined", lambda: extract_combined_best(
            solutions, node_list, graph, dual_vars)),
    ]

    for name, extract_fn in strategies:
        t0 = time.monotonic()
        all_is, breakdown = extract_fn()
        elapsed_ms = (time.monotonic() - t0) * 1000

        stats = compute_extraction_stats(
            all_is, dual_vars, breakdown, name, elapsed_ms
        )
        stats_list.append(stats)

        if verbose:
            print(f"  {name}: {stats.unique_is_count} unique, "
                  f"{stats.profitable_count} profitable (>1), "
                  f"{stats.relaxed_count} relaxed (>0.5), "
                  f"{elapsed_ms:.1f}ms")

    # Also test sub-IS extraction on best result
    best_base = max(stats_list, key=lambda s: s.unique_is_count)
    best_name = best_base.strategy_name
    best_extract_fn = [fn for n, fn in strategies if n == best_name][0]
    best_is, _ = best_extract_fn()

    t0 = time.monotonic()
    sub_is, sub_bd = extract_sub_is(best_is, graph, dual_vars, min_profit=0.5)
    elapsed_ms = (time.monotonic() - t0) * 1000

    sub_stats = compute_extraction_stats(
        sub_is, dual_vars, sub_bd, f"sub_is_from_{best_name}", elapsed_ms
    )
    stats_list.append(sub_stats)

    if verbose:
        print(f"  sub_is_from_{best_name}: {sub_stats.unique_is_count} unique, "
              f"{sub_stats.profitable_count} profitable, "
              f"{sub_stats.relaxed_count} relaxed")

    return stats_list


def run_cg_experiment(
    graph: nx.Graph,
    strategy: str,
    profitability_threshold: float = 1.0,
    max_iterations: int = 500,
    verbose: bool = False,
) -> ColumnGenerationStats:
    """Run full CG with a specific extraction strategy."""
    oracle = ExperimentalDiracOracle(
        extraction_strategy=strategy,
        profitability_threshold=profitability_threshold,
    )

    t0 = time.monotonic()
    num_colors, coloring, stats = column_generation(
        graph, oracle, max_iterations=max_iterations, verbose=verbose,
    )
    elapsed = time.monotonic() - t0

    valid = verify_coloring(graph, coloring) if coloring else False

    return ColumnGenerationStats(
        strategy_name=f"{strategy}_thresh{profitability_threshold}",
        chi=num_colors,
        valid=valid,
        iterations=stats.get("iterations", 0),
        columns_generated=stats.get("columns_generated", 0),
        wall_seconds=round(elapsed, 2),
    )


def run_full_experiment(
    n: int,
    p: float,
    seed: int = 42,
    max_iterations: int = 500,
    verbose: bool = False,
    run_cg: bool = True,
) -> ExperimentResult:
    """Run full experiment on an ER graph."""
    graph = erdos_renyi(n, p, seed=seed)
    graph_name = f"ER({n},{p})"

    print(f"\n{'='*60}")
    print(f"Experiment: {graph_name}  n={n} m={graph.number_of_edges()}")
    print(f"{'='*60}")

    # Greedy baseline
    greedy_coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
    greedy_chi = max(greedy_coloring.values()) + 1 if greedy_coloring else 0
    print(f"Greedy baseline: chi={greedy_chi}")

    result = ExperimentResult(
        graph_name=graph_name,
        n_nodes=n,
        n_edges=graph.number_of_edges(),
        greedy_chi=greedy_chi,
        num_dirac_samples=0,
    )

    if not DIRAC_AVAILABLE:
        print("Dirac not available, skipping extraction analysis")
        return result

    # Get initial dual vars from RMP with singletons
    from quantum_colgen.master_problem import solve_rmp
    singleton_cols = [frozenset([i]) for i in range(n)]
    _, dual_vars = solve_rmp(singleton_cols, n)

    if dual_vars is None:
        print("Failed to get initial duals, skipping")
        return result

    print(f"Initial dual vars: min={dual_vars.min():.3f} max={dual_vars.max():.3f} "
          f"mean={dual_vars.mean():.3f}")

    # Get Dirac solutions
    print("Fetching Dirac solutions...")
    oracle = ExperimentalDiracOracle(extraction_strategy="baseline")
    _ = oracle.solve(graph, dual_vars)

    if oracle.last_solutions is None:
        print("No Dirac solutions returned")
        return result

    solutions = oracle.last_solutions
    node_list = oracle.last_node_list
    result.num_dirac_samples = len(solutions)
    print(f"Got {len(solutions)} Dirac samples")

    # Analyze raw solutions
    result.raw_solution_analysis = analyze_solutions(solutions)

    # Run extraction analysis
    print("\nExtraction strategy comparison:")
    result.extraction_stats = run_extraction_analysis(
        graph, dual_vars, solutions, node_list, verbose=True
    )

    # Run CG experiments (optional, expensive)
    if run_cg:
        print("\nRunning CG experiments...")
        strategies_to_test = [
            ("baseline", 1.0),
            ("combined", 1.0),
            ("combined", 0.8),
            ("combined", 0.5),
        ]

        for strategy, threshold in strategies_to_test:
            print(f"  Testing {strategy} with threshold={threshold}...")
            try:
                cg_stats = run_cg_experiment(
                    graph, strategy, threshold,
                    max_iterations=max_iterations,
                    verbose=verbose,
                )
                result.cg_stats.append(cg_stats)
                valid_str = "VALID" if cg_stats.valid else "INVALID"
                print(f"    chi={cg_stats.chi} iters={cg_stats.iterations} "
                      f"cols={cg_stats.columns_generated} [{valid_str}]")
            except Exception as e:
                print(f"    ERROR: {e}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(result: ExperimentResult):
    """Print experiment summary."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nGraph: {result.graph_name}")
    print(f"Nodes: {result.n_nodes}, Edges: {result.n_edges}")
    print(f"Greedy chi: {result.greedy_chi}")
    print(f"Dirac samples: {result.num_dirac_samples}")

    if result.extraction_stats:
        print("\nExtraction Results:")
        print("-" * 70)
        print(f"{'Strategy':<25} {'Unique':<8} {'Profit>1':<10} {'Relax>0.5':<10} {'Time(ms)':<10}")
        print("-" * 70)
        for s in result.extraction_stats:
            print(f"{s.strategy_name:<25} {s.unique_is_count:<8} "
                  f"{s.profitable_count:<10} {s.relaxed_count:<10} "
                  f"{s.extraction_time_ms:<10.1f}")

    if result.cg_stats:
        print("\nColumn Generation Results:")
        print("-" * 70)
        print(f"{'Strategy':<30} {'chi':<6} {'Valid':<6} {'Iters':<8} {'Cols':<8} {'Time(s)':<8}")
        print("-" * 70)
        for s in result.cg_stats:
            valid_str = "Y" if s.valid else "N"
            chi_str = str(s.chi) if s.chi is not None else "-"
            print(f"{s.strategy_name:<30} {chi_str:<6} {valid_str:<6} "
                  f"{s.iterations:<8} {s.columns_generated:<8} {s.wall_seconds:<8.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiments to improve Dirac IS extraction"
    )
    parser.add_argument("--nodes", "-n", type=int, default=15)
    parser.add_argument("--edge-prob", "-p", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--save", type=str, help="Save results to JSON")
    parser.add_argument("--all", action="store_true", help="Run CG experiments too")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick analysis only (no CG runs)"
    )

    args = parser.parse_args()

    run_cg = args.all and not args.quick

    result = run_full_experiment(
        n=args.nodes,
        p=args.edge_prob,
        seed=args.seed,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        run_cg=run_cg,
    )

    print_summary(result)

    if args.save:
        output = asdict(result)
        with open(args.save, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.save}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
