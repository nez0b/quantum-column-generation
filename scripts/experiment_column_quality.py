#!/usr/bin/env python
"""Column quality experiments: 5 novel ideas for better Dirac IS extraction.

Tests five ideas for extracting more/better profitable independent sets from
Dirac-3 QP solutions in the column generation pricing subproblem:

  E1: Dual-perturbed multi-solve (perturb dual weights → different QP landscapes)
  E2: Subgraph decomposition (partition into communities → smaller QPs)
  E3: Solution vector clustering (k-means on Dirac solutions → centroids + extremes)
  E4: 2-swap local search (extend 1-swap → escape local optima)
  E5: Column memory / diversity pressure (penalize over-represented nodes in QP)

Usage:
    source ~/.zshrc   # Load QCI_TOKEN

    # Run all experiments on default graphs
    uv run python scripts/experiment_column_quality.py --all

    # Run specific experiments
    uv run python scripts/experiment_column_quality.py --experiments E3 E4

    # Quick test on small graph
    uv run python scripts/experiment_column_quality.py --quick --er-sizes 30

    # Custom graphs and parameters
    uv run python scripts/experiment_column_quality.py --er-sizes 30 40 50 --er-prob 0.5

See docs/plans/2026-02-06-column-quality-design.md for experiment design.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from quantum_colgen.graphs import erdos_renyi
from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.master_problem import solve_rmp, solve_final_ilp
from quantum_colgen.pricing.base import PricingOracle
from quantum_colgen.timing import OracleTimer

try:
    from quantum_colgen.pricing.dirac_oracle import (
        DiracPricingOracle,
        _construct_gibbons_matrix,
        _local_search,
        _local_search_2swap,
        _cluster_solutions,
        _greedy_prune_dual_desc,
        _greedy_prune_dual_asc,
        _greedy_prune_random,
        DIRAC_AVAILABLE,
    )
    from eqc_models.solvers import Dirac3ContinuousCloudSolver
    from eqc_models.base import QuadraticModel
except ImportError:
    DIRAC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data structures for experiment results
# ---------------------------------------------------------------------------

@dataclass
class IterationMetrics:
    """Metrics from a single CG iteration."""
    iteration: int
    rmp_obj: float
    unique_profitable_cols: int
    total_dual_weight: float  # sum of dual weights across all returned IS
    avg_jaccard_distance: float  # pairwise diversity
    extract_time_ms: float
    api_time_ms: float


@dataclass
class ExperimentResult:
    """Full result from one experiment configuration on one graph."""
    experiment_name: str
    graph_name: str
    n_nodes: int
    n_edges: int
    params: Dict[str, Any]

    # Final results
    chi: Optional[int] = None
    valid: bool = False
    total_iterations: int = 0
    total_columns: int = 0
    total_api_calls: int = 0
    wall_seconds: float = 0.0

    # Per-iteration trace (first 5 + last 5)
    iteration_trace: List[Dict[str, Any]] = field(default_factory=list)

    # Comparison to baseline
    chi_delta: Optional[int] = None  # chi - baseline_chi
    column_ratio: Optional[float] = None  # our_cols / baseline_cols


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def jaccard_distance(a: Set[int], b: Set[int]) -> float:
    """Jaccard distance between two sets (1 - Jaccard similarity)."""
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def compute_column_diversity(columns: List[Set[int]]) -> float:
    """Average pairwise Jaccard distance among columns."""
    if len(columns) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            total += jaccard_distance(columns[i], columns[j])
            count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Instrumented CG loop — shared by all experiments
# ---------------------------------------------------------------------------

def run_instrumented_cg(
    graph: nx.Graph,
    oracle: PricingOracle,
    max_iterations: int = 500,
    verbose: bool = False,
) -> ExperimentResult:
    """Run column generation with per-iteration instrumentation.

    Returns an ExperimentResult with iteration-level metrics.
    """
    node_list = sorted(graph.nodes())
    num_vertices = len(node_list)

    columns: List[FrozenSet[int]] = [frozenset([i]) for i in range(num_vertices)]
    known_sigs = {tuple(sorted(c)) for c in columns}

    trace = []
    t_start = time.monotonic()

    for iteration in range(1, max_iterations + 1):
        obj, dual_vars = solve_rmp(columns, num_vertices)
        if dual_vars is None:
            break

        t_api = time.monotonic()
        new_cols = oracle.solve(graph, dual_vars)
        api_ms = (time.monotonic() - t_api) * 1000

        if not new_cols:
            break

        # Compute iteration metrics
        added_cols = []
        for col_set in new_cols:
            sig = tuple(sorted(col_set))
            if sig not in known_sigs:
                columns.append(frozenset(col_set))
                known_sigs.add(sig)
                added_cols.append(col_set)

        if not added_cols:
            break

        total_dual = sum(
            sum(dual_vars[v] for v in col) for col in added_cols
        )
        diversity = compute_column_diversity(added_cols) if len(added_cols) > 1 else 0.0

        # Record in trace (keep first 5 and last 5)
        iter_data = {
            "iteration": iteration,
            "rmp_obj": round(obj, 4),
            "new_cols": len(added_cols),
            "total_cols": len(columns),
            "total_dual_weight": round(total_dual, 4),
            "diversity": round(diversity, 4),
            "api_ms": round(api_ms, 1),
        }
        trace.append(iter_data)

        if verbose:
            print(f"  Iter {iteration}: obj={obj:.3f} +{len(added_cols)} cols "
                  f"(total {len(columns)}) diversity={diversity:.3f}")

    wall_s = time.monotonic() - t_start

    # Final ILP
    num_colors, selected_indices = solve_final_ilp(columns, num_vertices)
    valid = False
    if num_colors is not None:
        coloring = [columns[i] for i in selected_indices]
        valid = verify_coloring(graph, coloring)

    # Trim trace to first 5 + last 5
    if len(trace) > 10:
        trimmed = trace[:5] + trace[-5:]
    else:
        trimmed = trace

    result = ExperimentResult(
        experiment_name="",
        graph_name="",
        n_nodes=num_vertices,
        n_edges=graph.number_of_edges(),
        params={},
        chi=num_colors,
        valid=valid,
        total_iterations=len(trace),
        total_columns=len(columns),
        total_api_calls=oracle.timer.num_calls if hasattr(oracle, 'timer') else 0,
        wall_seconds=round(wall_s, 2),
        iteration_trace=trimmed,
    )
    return result


# ===================================================================
# EXPERIMENT ORACLES
# ===================================================================

# ---------------------------------------------------------------------------
# E0: Baseline oracle (production config)
# ---------------------------------------------------------------------------

def make_baseline_oracle(seed: int = 42) -> DiracPricingOracle:
    """Create production baseline Dirac oracle."""
    return DiracPricingOracle(
        method="gibbons",
        num_samples=100,
        multi_prune=True,
        randomized_rounding=True,
        num_random_rounds=10,
        random_seed=seed,
        local_search_passes=5,
    )


# ---------------------------------------------------------------------------
# E1: Dual-Perturbed Multi-Solve
# ---------------------------------------------------------------------------

class DualPerturbedOracle(PricingOracle):
    """Solve multiple QPs per CG iteration with perturbed dual weights.

    Perturbs dual_vars as: w'[i] = w[i] * (1 + epsilon * N(0,1))
    then calls the base oracle on each perturbed version and unions results.
    """

    def __init__(
        self,
        epsilon: float = 0.10,
        num_perturbations: int = 2,
        seed: int = 42,
        **base_kwargs,
    ):
        self.epsilon = epsilon
        self.num_perturbations = num_perturbations
        self._rng = np.random.RandomState(seed)
        self.timer = OracleTimer()

        # Create base oracles for each perturbation + the original
        self._base_kwargs = {
            "method": "gibbons",
            "num_samples": 100,
            "multi_prune": True,
            "randomized_rounding": True,
            "num_random_rounds": 10,
            "local_search_passes": 5,
            "random_seed": seed,
        }
        self._base_kwargs.update(base_kwargs)

    def _solve_qp_with_weights(
        self, graph: nx.Graph, weights: np.ndarray,
    ) -> Tuple[Optional[list], List[int], float]:
        """Submit a Gibbons QP with given weights and return raw solutions.

        Returns (solutions, node_list, api_seconds). Solutions is None on failure.
        """
        pos_nodes = [v for v in graph.nodes() if weights[v] > 1e-5]
        if not pos_nodes:
            return None, [], 0.0

        subgraph = graph.subgraph(pos_nodes)
        if not subgraph.edges():
            return None, sorted(pos_nodes), 0.0

        w = {v: float(weights[v]) for v in pos_nodes}
        complement = nx.complement(subgraph)
        B = _construct_gibbons_matrix(complement, w)

        n = B.shape[0]
        C = np.zeros(n, dtype=np.float64)
        J = B.astype(np.float64)

        model = QuadraticModel(C, J)
        model.upper_bound = np.ones(n, dtype=np.float64)
        solver = Dirac3ContinuousCloudSolver()

        t0 = time.monotonic()
        response = solver.solve(
            model, sum_constraint=1,
            num_samples=self._base_kwargs.get("num_samples", 100),
            relaxation_schedule=2,
        )
        api_s = time.monotonic() - t0

        if not (response and "results" in response and "solutions" in response["results"]):
            return None, sorted(pos_nodes), api_s

        raw = response["results"]["solutions"]
        if not raw:
            return None, sorted(pos_nodes), api_s

        solutions = [np.array(s, dtype=np.float64) for s in raw]
        return solutions, sorted(subgraph.nodes()), api_s

    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        # Collect raw Dirac solutions from original + perturbed QPs
        all_solutions: List[np.ndarray] = []
        node_list = None
        total_api_s = 0.0

        # Original (unperturbed) solve
        sols, nl, api_s = self._solve_qp_with_weights(graph, dual_vars)
        total_api_s += api_s
        if sols:
            all_solutions.extend(sols)
            node_list = nl

        # Perturbed solves — different QP landscapes, same extraction
        for _ in range(self.num_perturbations):
            noise = self._rng.randn(len(dual_vars))
            perturbed = dual_vars * (1.0 + self.epsilon * noise)
            perturbed = np.maximum(perturbed, 0.0)

            sols, nl, api_s = self._solve_qp_with_weights(graph, perturbed)
            total_api_s += api_s
            if sols:
                all_solutions.extend(sols)
                if node_list is None:
                    node_list = nl

        if not all_solutions or node_list is None:
            self.timer.record(api_seconds=total_api_s)
            return []

        # Extract all IS using ORIGINAL duals (not perturbed)
        t0 = time.monotonic()
        base_oracle = DiracPricingOracle(**self._base_kwargs)
        result = base_oracle._extract_independent_sets(
            all_solutions, node_list, graph, dual_vars
        )
        extract_s = time.monotonic() - t0

        self.timer.record(
            api_seconds=total_api_s,
            extract_seconds=extract_s,
            columns_found=len(result),
        )
        return result


# ---------------------------------------------------------------------------
# E2: Subgraph Decomposition
# ---------------------------------------------------------------------------

class SubgraphDecompOracle(PricingOracle):
    """Decompose into overlapping community subgraphs, solve each separately.

    Uses greedy modularity to find communities, adds overlap, and solves
    separate QPs per partition.
    """

    def __init__(
        self,
        num_clusters: int = 2,
        overlap_fraction: float = 0.1,
        seed: int = 42,
        **base_kwargs,
    ):
        self.num_clusters = num_clusters
        self.overlap_fraction = overlap_fraction
        self.seed = seed
        self.timer = OracleTimer()

        self._base_kwargs = {
            "method": "gibbons",
            "num_samples": 100,
            "multi_prune": True,
            "randomized_rounding": True,
            "num_random_rounds": 10,
            "local_search_passes": 5,
            "random_seed": seed,
        }
        self._base_kwargs.update(base_kwargs)

    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        pos_nodes = [v for v in graph.nodes() if dual_vars[v] > 1e-5]
        if not pos_nodes:
            return []

        subgraph = graph.subgraph(pos_nodes)
        if len(pos_nodes) < 10:
            # Too small to decompose — fall back to standard
            oracle = DiracPricingOracle(**self._base_kwargs)
            cols = oracle.solve(graph, dual_vars)
            self.timer.record(
                api_seconds=oracle.timer.total_api_seconds,
                columns_found=len(cols),
            )
            return cols

        # Find communities via greedy modularity
        try:
            communities = list(nx.community.greedy_modularity_communities(
                subgraph, resolution=1.0
            ))
        except Exception:
            communities = [set(pos_nodes)]

        # Merge small communities until we have num_clusters
        while len(communities) > self.num_clusters:
            # Merge two smallest
            communities.sort(key=len)
            merged = communities[0] | communities[1]
            communities = [merged] + communities[2:]

        # Add overlap: for each community, add neighbors from other communities
        overlap_size = max(1, int(self.overlap_fraction * len(pos_nodes)))
        expanded_communities = []
        for comm in communities:
            expanded = set(comm)
            boundary = set()
            for v in comm:
                for u in subgraph.neighbors(v):
                    if u not in comm:
                        boundary.add(u)
            # Add highest-dual boundary nodes
            boundary_sorted = sorted(boundary, key=lambda v: dual_vars[v], reverse=True)
            expanded.update(boundary_sorted[:overlap_size])
            expanded_communities.append(expanded)

        # Solve each partition
        seen: set = set()
        all_cols: List[Set[int]] = []
        total_api_s = 0.0

        for comm_nodes in expanded_communities:
            if len(comm_nodes) < 3:
                continue

            oracle = DiracPricingOracle(**self._base_kwargs)
            # Create subgraph for this community
            comm_subgraph = graph.subgraph(comm_nodes)
            t0 = time.monotonic()
            cols = oracle.solve(comm_subgraph, dual_vars)
            total_api_s += time.monotonic() - t0

            # IS in induced subgraph is automatically IS in full graph.
            # oracle.solve already checks profitability, just deduplicate.
            for col in cols:
                sig = frozenset(col)
                if sig not in seen:
                    seen.add(sig)
                    all_cols.append(col)

        self.timer.record(api_seconds=total_api_s, columns_found=len(all_cols))
        return all_cols


# ---------------------------------------------------------------------------
# E3: Solution Vector Clustering
# ---------------------------------------------------------------------------

class ClusteringOracle(PricingOracle):
    """Extract IS from clustered Dirac solution vectors.

    Clusters ~100 solutions via k-means, then extracts from:
    - centroids (denoised consensus)
    - extremes (maximum diversity)
    - or both
    """

    def __init__(
        self,
        k: int = 5,
        extract_from: str = "both",  # "centroid", "extreme", "both"
        seed: int = 42,
        **base_kwargs,
    ):
        if not DIRAC_AVAILABLE:
            raise ImportError("Dirac not available")

        self.k = k
        self.extract_from = extract_from
        self.timer = OracleTimer()
        self._rng = random.Random(seed)
        self._seed = seed

        self._base_kwargs = {
            "method": "gibbons",
            "num_samples": 100,
            "multi_prune": True,
            "randomized_rounding": True,
            "num_random_rounds": 10,
            "local_search_passes": 5,
            "random_seed": seed,
        }
        self._base_kwargs.update(base_kwargs)

    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        pos_nodes = [v for v in graph.nodes() if dual_vars[v] > 1e-5]
        if not pos_nodes:
            return []

        subgraph = graph.subgraph(pos_nodes)
        if not subgraph.edges():
            total = sum(dual_vars[v] for v in pos_nodes)
            return [set(pos_nodes)] if total > 1 + 1e-5 else []

        # Build Gibbons matrix and solve (same as standard oracle)
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
            num_samples=self._base_kwargs.get("num_samples", 100),
            relaxation_schedule=2,
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

        # Cluster solutions
        centroids, extremes = _cluster_solutions(solutions, k=self.k, seed=self._seed)

        # Choose which to extract from
        if self.extract_from == "centroid":
            extract_solutions = centroids
        elif self.extract_from == "extreme":
            extract_solutions = extremes
        else:  # "both"
            extract_solutions = centroids + extremes

        # Extract ONLY from clustered representatives to isolate the effect.
        # (The baseline already extracts from all 100 — adding them here would
        # conflate the clustering effect with the standard extraction.)

        # Use standard extraction on the clustered solution list
        t0 = time.monotonic()
        base_oracle = DiracPricingOracle(**self._base_kwargs)
        result = base_oracle._extract_independent_sets(
            extract_solutions, node_list, graph, dual_vars
        )
        extract_seconds = time.monotonic() - t0

        self.timer.record(
            api_seconds=api_seconds,
            extract_seconds=extract_seconds,
            columns_found=len(result),
        )
        return result


# ---------------------------------------------------------------------------
# E4: 2-Swap Local Search Oracle
# ---------------------------------------------------------------------------

class TwoSwapOracle(PricingOracle):
    """Dirac oracle with 2-swap local search instead of (or after) 1-swap.

    After standard extraction + 1-swap, applies 2-swap to each IS for
    further refinement. More expensive but escapes local optima.
    """

    def __init__(
        self,
        max_2swap_passes: int = 3,
        seed: int = 42,
        **base_kwargs,
    ):
        if not DIRAC_AVAILABLE:
            raise ImportError("Dirac not available")

        self.max_2swap_passes = max_2swap_passes
        self.timer = OracleTimer()
        self._seed = seed

        self._base_kwargs = {
            "method": "gibbons",
            "num_samples": 100,
            "multi_prune": True,
            "randomized_rounding": True,
            "num_random_rounds": 10,
            "local_search_passes": 5,
            "random_seed": seed,
        }
        self._base_kwargs.update(base_kwargs)

    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        # Run standard oracle first
        base_oracle = DiracPricingOracle(**self._base_kwargs)
        base_cols = base_oracle.solve(graph, dual_vars)

        api_s = base_oracle.timer.total_api_seconds
        t0 = time.monotonic()

        # Apply 2-swap to each IS
        seen: set = set()
        refined_cols: List[Set[int]] = []

        # Keep originals
        for col in base_cols:
            sig = frozenset(col)
            if sig not in seen:
                seen.add(sig)
                refined_cols.append(col)

        # Apply 2-swap and add any new IS discovered
        for col in base_cols:
            improved = _local_search_2swap(
                graph, col, dual_vars, max_passes=self.max_2swap_passes
            )
            sig = frozenset(improved)
            if sig not in seen:
                profit = sum(dual_vars[v] for v in improved)
                if profit > 1 + 1e-5:
                    seen.add(sig)
                    refined_cols.append(set(improved))

        extract_s = time.monotonic() - t0

        self.timer.record(
            api_seconds=api_s,
            extract_seconds=extract_s,
            columns_found=len(refined_cols),
        )
        return refined_cols


# ---------------------------------------------------------------------------
# E5: Column Memory / Diversity Pressure
# ---------------------------------------------------------------------------

class ColumnMemoryOracle(PricingOracle):
    """Dirac oracle with diversity pressure on over-represented nodes.

    Tracks which nodes appear frequently in previously found columns and
    penalizes them by reducing their effective weight before building the
    Gibbons matrix:
        w'[i] = w[i] / (1 + lambda * frequency[i])
    This correctly shifts the QP landscape to disfavor over-represented nodes
    while preserving the Gibbons matrix structure.
    """

    def __init__(
        self,
        lambda_penalty: float = 0.5,
        seed: int = 42,
        **base_kwargs,
    ):
        if not DIRAC_AVAILABLE:
            raise ImportError("Dirac not available")

        self.lambda_penalty = lambda_penalty
        self.timer = OracleTimer()
        self._rng = random.Random(seed)
        self._seed = seed

        self._base_kwargs = {
            "method": "gibbons",
            "num_samples": 100,
            "multi_prune": True,
            "randomized_rounding": True,
            "num_random_rounds": 10,
            "local_search_passes": 5,
            "random_seed": seed,
        }
        self._base_kwargs.update(base_kwargs)

        # Column memory: track node frequency across iterations
        self._column_history: List[Set[int]] = []
        self._node_frequency: Dict[int, float] = defaultdict(float)

    def _update_memory(self, columns: List[Set[int]]) -> None:
        """Update node frequency from newly found columns."""
        self._column_history.extend(columns)
        # Recompute frequencies
        self._node_frequency.clear()
        total = len(self._column_history) if self._column_history else 1
        for col in self._column_history:
            for v in col:
                self._node_frequency[v] += 1.0 / total

    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        pos_nodes = [v for v in graph.nodes() if dual_vars[v] > 1e-5]
        if not pos_nodes:
            return []

        subgraph = graph.subgraph(pos_nodes)
        if not subgraph.edges():
            total = sum(dual_vars[v] for v in pos_nodes)
            return [set(pos_nodes)] if total > 1 + 1e-5 else []

        # Build Gibbons matrix with diversity-adjusted weights.
        # Reduce effective weight of over-represented nodes so Dirac
        # explores under-represented regions of the IS space.
        adjusted_weights = {
            v: float(dual_vars[v]) / (1.0 + self.lambda_penalty * self._node_frequency.get(v, 0.0))
            for v in pos_nodes
        }
        complement = nx.complement(subgraph)
        B = _construct_gibbons_matrix(complement, adjusted_weights)
        node_list = sorted(subgraph.nodes())

        # Solve with diversity-adjusted matrix
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
            num_samples=self._base_kwargs.get("num_samples", 100),
            relaxation_schedule=2,
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

        # Extract using standard pipeline (profitability checked against ORIGINAL duals)
        t0_ext = time.monotonic()
        base_oracle = DiracPricingOracle(**self._base_kwargs)
        result = base_oracle._extract_independent_sets(
            solutions, node_list, graph, dual_vars
        )
        extract_seconds = time.monotonic() - t0_ext

        # Update memory with found columns
        self._update_memory(result)

        self.timer.record(
            api_seconds=api_seconds,
            extract_seconds=extract_seconds,
            columns_found=len(result),
        )
        return result


# ===================================================================
# EXPERIMENT RUNNER
# ===================================================================

@dataclass
class GraphSpec:
    """Specification for a test graph."""
    n: int
    p: float
    seed: int = 42

    @property
    def name(self) -> str:
        return f"ER({self.n},{self.p})"

    def generate(self) -> nx.Graph:
        return erdos_renyi(self.n, self.p, seed=self.seed)


@dataclass
class ExperimentConfig:
    """Configuration for one experiment run."""
    name: str
    make_oracle: Any  # callable() -> PricingOracle
    params: Dict[str, Any]


def build_experiment_configs(
    experiments: List[str],
    seed: int = 42,
) -> List[ExperimentConfig]:
    """Build experiment configurations for the requested experiment IDs."""
    configs = []

    # E0: Baseline (always included)
    configs.append(ExperimentConfig(
        name="E0_baseline",
        make_oracle=lambda: make_baseline_oracle(seed),
        params={"type": "baseline"},
    ))

    if "E1" in experiments or "all" in experiments:
        for eps in [0.05, 0.10, 0.15]:
            for n_pert in [2, 3]:
                configs.append(ExperimentConfig(
                    name=f"E1_dual_perturb_eps{eps}_n{n_pert}",
                    make_oracle=lambda e=eps, n=n_pert: DualPerturbedOracle(
                        epsilon=e, num_perturbations=n, seed=seed,
                    ),
                    params={"epsilon": eps, "num_perturbations": n_pert},
                ))

    if "E2" in experiments or "all" in experiments:
        for nc in [2, 3]:
            for overlap in [0.1, 0.2]:
                configs.append(ExperimentConfig(
                    name=f"E2_subgraph_c{nc}_o{overlap}",
                    make_oracle=lambda nc_=nc, ov=overlap: SubgraphDecompOracle(
                        num_clusters=nc_, overlap_fraction=ov, seed=seed,
                    ),
                    params={"num_clusters": nc, "overlap_fraction": overlap},
                ))

    if "E3" in experiments or "all" in experiments:
        for k in [5, 10]:
            for extract in ["centroid", "extreme", "both"]:
                configs.append(ExperimentConfig(
                    name=f"E3_cluster_k{k}_{extract}",
                    make_oracle=lambda k_=k, ex=extract: ClusteringOracle(
                        k=k_, extract_from=ex, seed=seed,
                    ),
                    params={"k": k, "extract_from": extract},
                ))

    if "E4" in experiments or "all" in experiments:
        for passes in [3, 5]:
            configs.append(ExperimentConfig(
                name=f"E4_2swap_p{passes}",
                make_oracle=lambda p=passes: TwoSwapOracle(
                    max_2swap_passes=p, seed=seed,
                ),
                params={"max_swap_passes": passes},
            ))

    if "E5" in experiments or "all" in experiments:
        for lam in [0.1, 0.5, 1.0]:
            configs.append(ExperimentConfig(
                name=f"E5_colmem_lam{lam}",
                make_oracle=lambda l=lam: ColumnMemoryOracle(
                    lambda_penalty=l, seed=seed,
                ),
                params={"lambda_penalty": lam},
            ))

    return configs


def run_experiment(
    graph: nx.Graph,
    graph_name: str,
    config: ExperimentConfig,
    max_iterations: int = 500,
    verbose: bool = False,
) -> ExperimentResult:
    """Run a single experiment configuration on a graph."""
    oracle = config.make_oracle()
    result = run_instrumented_cg(
        graph, oracle, max_iterations=max_iterations, verbose=verbose,
    )
    result.experiment_name = config.name
    result.graph_name = graph_name
    result.params = config.params
    return result


def run_all_experiments(
    graphs: List[GraphSpec],
    experiments: List[str],
    max_iterations: int = 500,
    seed: int = 42,
    verbose: bool = False,
    output_dir: str = "results/column_quality_experiments",
) -> List[ExperimentResult]:
    """Run all experiment configurations on all graphs."""
    os.makedirs(output_dir, exist_ok=True)
    configs = build_experiment_configs(experiments, seed=seed)

    all_results: List[ExperimentResult] = []

    for gs in graphs:
        graph = gs.generate()
        greedy = nx.coloring.greedy_color(graph, strategy="largest_first")
        greedy_chi = max(greedy.values()) + 1 if greedy else 0

        print(f"\n{'='*70}")
        print(f"Graph: {gs.name}  n={gs.n}  m={graph.number_of_edges()}  "
              f"greedy_chi={greedy_chi}")
        print(f"{'='*70}")

        baseline_result = None

        for config in configs:
            print(f"\n  Running {config.name}...")
            try:
                result = run_experiment(
                    graph, gs.name, config,
                    max_iterations=max_iterations, verbose=verbose,
                )

                # Track baseline for comparison
                if config.name == "E0_baseline":
                    baseline_result = result

                # Compute deltas vs baseline
                if baseline_result is not None and config.name != "E0_baseline":
                    if result.chi is not None and baseline_result.chi is not None:
                        result.chi_delta = result.chi - baseline_result.chi
                    if baseline_result.total_columns > 0:
                        result.column_ratio = round(
                            result.total_columns / baseline_result.total_columns, 2
                        )

                valid_str = "VALID" if result.valid else "INVALID"
                chi_str = str(result.chi) if result.chi is not None else "-"
                delta_str = ""
                if result.chi_delta is not None:
                    delta_str = f" (delta={result.chi_delta:+d})"

                print(f"    chi={chi_str}{delta_str}  iters={result.total_iterations}  "
                      f"cols={result.total_columns}  "
                      f"api_calls={result.total_api_calls}  "
                      f"time={result.wall_seconds:.1f}s [{valid_str}]")

                all_results.append(result)

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

        # Save per-graph results
        graph_results = [r for r in all_results if r.graph_name == gs.name]
        output_file = os.path.join(output_dir, f"{gs.name.replace(',', '_').replace('(', '').replace(')', '')}.json")
        with open(output_file, "w") as f:
            json.dump([asdict(r) for r in graph_results], f, indent=2)
        print(f"\n  Saved {len(graph_results)} results to {output_file}")

    return all_results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(results: List[ExperimentResult]) -> None:
    """Print a compact summary table of all results."""
    print("\n" + "=" * 90)
    print("EXPERIMENT SUMMARY")
    print("=" * 90)

    # Group by graph
    by_graph: Dict[str, List[ExperimentResult]] = defaultdict(list)
    for r in results:
        by_graph[r.graph_name].append(r)

    for graph_name, graph_results in by_graph.items():
        print(f"\n--- {graph_name} ---")
        print(f"{'Experiment':<35} {'chi':>4} {'delta':>6} {'iters':>6} "
              f"{'cols':>6} {'API':>4} {'time(s)':>8} {'valid':>5}")
        print("-" * 80)

        for r in graph_results:
            chi_str = str(r.chi) if r.chi is not None else "-"
            delta_str = f"{r.chi_delta:+d}" if r.chi_delta is not None else "-"
            valid_str = "Y" if r.valid else "N"
            print(f"{r.experiment_name:<35} {chi_str:>4} {delta_str:>6} "
                  f"{r.total_iterations:>6} {r.total_columns:>6} "
                  f"{r.total_api_calls:>4} {r.wall_seconds:>8.1f} {valid_str:>5}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Column quality experiments for Dirac IS extraction",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=["all"],
        help="Experiment IDs to run: E1 E2 E3 E4 E5 or 'all'",
    )
    parser.add_argument("--er-sizes", nargs="+", type=int, default=[30, 40, 50])
    parser.add_argument("--er-prob", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--output-dir", default="results/column_quality_experiments",
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: only E3 and E4 (no extra API calls)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all experiments (overrides --experiments)",
    )

    args = parser.parse_args()

    if not DIRAC_AVAILABLE:
        print("ERROR: Dirac dependencies not available.")
        print("Install with: uv pip install qci-client eqc-models")
        return 1

    # Build graph specs
    graphs = [GraphSpec(n=n, p=args.er_prob, seed=args.seed) for n in args.er_sizes]

    # Determine experiments
    if args.all:
        experiments = ["all"]
    elif args.quick:
        experiments = ["E3", "E4"]
    else:
        experiments = args.experiments

    print(f"Experiments: {experiments}")
    print(f"Graphs: {[g.name for g in graphs]}")
    print(f"Max iterations: {args.max_iterations}")

    results = run_all_experiments(
        graphs=graphs,
        experiments=experiments,
        max_iterations=args.max_iterations,
        seed=args.seed,
        verbose=args.verbose,
        output_dir=args.output_dir,
    )

    print_summary(results)

    # Save combined results
    combined_file = os.path.join(args.output_dir, "all_results.json")
    with open(combined_file, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nAll results saved to {combined_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
