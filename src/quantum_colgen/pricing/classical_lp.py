"""Classical LP-relaxation pricing oracle with extraction.

Uses LP relaxation of MWIS (polynomial time) instead of exact MILP (NP-hard).
Extracts multiple independent sets from the fractional LP solution using
thresholding and pruning, similar to the Dirac oracle approach.

This provides a classical baseline that's directly comparable to Dirac:
both use continuous relaxation + extraction, allowing us to isolate
whether Dirac's value comes from quantum annealing or from the
continuous relaxation approach itself.
"""

import random
import time
from typing import List, Optional, Set

import networkx as nx
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .base import PricingOracle
from ..timing import OracleTimer


# ---------------------------------------------------------------------------
# Pruning strategies (same as dirac_oracle.py)
# ---------------------------------------------------------------------------

def _greedy_prune_dual_desc(
    support: Set[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
) -> Set[int]:
    """Greedy prune: highest dual weight first."""
    pruned: Set[int] = set()
    for v in sorted(support, key=lambda v: dual_vars[v], reverse=True):
        if not any(graph.has_edge(v, u) for u in pruned):
            pruned.add(v)
    return pruned


def _greedy_prune_dual_asc(
    support: Set[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
) -> Set[int]:
    """Greedy prune: lowest dual weight first."""
    pruned: Set[int] = set()
    for v in sorted(support, key=lambda v: dual_vars[v], reverse=False):
        if not any(graph.has_edge(v, u) for u in pruned):
            pruned.add(v)
    return pruned


def _greedy_prune_random(
    support: Set[int],
    graph: nx.Graph,
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


def _local_search(
    graph: nx.Graph,
    independent_set: Set[int],
    dual_vars: np.ndarray,
    max_passes: int = 5,
) -> Set[int]:
    """Improve an independent set via 1-swap local search."""
    improved = set(independent_set)
    candidates = sorted(
        [v for v in graph.nodes() if dual_vars[v] > 1e-8],
        key=lambda v: dual_vars[v],
        reverse=True,
    )

    for _ in range(max_passes):
        changed = False
        for v in candidates:
            if v in improved:
                continue
            conflicts = {u for u in improved if graph.has_edge(v, u)}
            gain = dual_vars[v] - sum(dual_vars[u] for u in conflicts)
            if gain > 1e-8:
                improved -= conflicts
                improved.add(v)
                changed = True
        if not changed:
            break
    return improved


class ClassicalLPPricingOracle(PricingOracle):
    """LP-relaxation pricing oracle for MWIS with multi-column extraction.

    Instead of solving the exact MILP (NP-hard), this oracle:
    1. Solves the LP relaxation of MWIS (polynomial time)
    2. Extracts multiple independent sets via thresholding + pruning

    This is directly comparable to the Dirac oracle, which also uses
    continuous optimization (Motzkin-Straus QP) + extraction.

    LP relaxation of MWIS:
        maximize  Σ w_i * x_i
        s.t.      x_u + x_v ≤ 1  for all edges (u,v)
                  0 ≤ x_i ≤ 1

    Parameters
    ----------
    support_thresholds : list of float
        Thresholds for extracting support sets from LP solution.
        Default: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    local_search_passes : int
        Number of 1-swap local search passes per extracted IS.
    multi_prune : bool
        If True, try multiple pruning strategies per support set.
    num_random_prune_trials : int
        Number of random pruning trials when multi_prune=True.
    randomized_rounding : bool
        If True, also use probabilistic extraction based on LP values.
    num_random_rounds : int
        Number of randomized rounding trials per LP solution.
    random_seed : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        support_thresholds: Optional[List[float]] = None,
        local_search_passes: int = 5,
        multi_prune: bool = True,
        num_random_prune_trials: int = 3,
        randomized_rounding: bool = True,
        num_random_rounds: int = 10,
        random_seed: Optional[int] = None,
    ):
        if support_thresholds is not None:
            self.support_thresholds = support_thresholds
        else:
            # LP relaxation often gives half-integral solutions (0.5)
            # Include thresholds below 0.5 to capture these
            self.support_thresholds = [0.1, 0.2, 0.3, 0.4, 0.49, 0.6, 0.7, 0.8, 0.9]

        self.local_search_passes = local_search_passes
        self.multi_prune = multi_prune
        self.num_random_prune_trials = num_random_prune_trials
        self.randomized_rounding = randomized_rounding
        self.num_random_rounds = num_random_rounds
        self._rng = random.Random(random_seed)

        self.timer = OracleTimer()

    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        """Solve pricing subproblem using LP relaxation + extraction."""
        node_list = sorted(graph.nodes())
        num_vertices = len(node_list)

        if num_vertices == 0:
            return []

        # Filter to positive-dual subgraph
        pos_indices = [i for i in range(num_vertices) if dual_vars[i] > 1e-10]
        if not pos_indices:
            return []

        filtered_nodes = [node_list[i] for i in pos_indices]
        filtered_weights = dual_vars[pos_indices]
        filtered_node_to_idx = {node: i for i, node in enumerate(filtered_nodes)}
        subgraph = graph.subgraph(filtered_nodes)

        n_filt = len(filtered_nodes)

        # If no edges, all positive-dual nodes form an IS
        if subgraph.number_of_edges() == 0:
            total = sum(dual_vars[v] for v in filtered_nodes)
            if total > 1 + 1e-5:
                self.timer.record(api_seconds=0, columns_found=1)
                return [set(filtered_nodes)]
            return []

        # Solve LP relaxation
        t0 = time.monotonic()
        lp_solution = self._solve_lp(subgraph, filtered_weights, filtered_node_to_idx)
        api_seconds = time.monotonic() - t0

        if lp_solution is None:
            self.timer.record(api_seconds=api_seconds)
            return []

        # Extract independent sets from LP solution
        t0 = time.monotonic()
        result = self._extract_independent_sets(
            lp_solution, filtered_nodes, graph, dual_vars
        )
        extract_seconds = time.monotonic() - t0

        self.timer.record(
            api_seconds=api_seconds,
            extract_seconds=extract_seconds,
            columns_found=len(result),
        )
        return result

    def _solve_lp(
        self,
        subgraph: nx.Graph,
        weights: np.ndarray,
        node_to_idx: dict,
    ) -> Optional[np.ndarray]:
        """Solve LP relaxation of MWIS."""
        n = len(weights)
        edges = list(subgraph.edges())

        # Objective: maximize sum(w_i * x_i) -> minimize -sum(w_i * x_i)
        c = -weights

        # Edge constraints: x_u + x_v <= 1
        if edges:
            A_ub = lil_matrix((len(edges), n), dtype=float)
            for i, (u, v) in enumerate(edges):
                A_ub[i, node_to_idx[u]] = 1
                A_ub[i, node_to_idx[v]] = 1
            b_ub = np.ones(len(edges))
        else:
            A_ub = None
            b_ub = None

        # Bounds: 0 <= x_i <= 1
        bounds = [(0, 1)] * n

        result = linprog(
            c,
            A_ub=A_ub.toarray() if A_ub is not None else None,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )

        if not result.success:
            return None

        return result.x

    def _extract_independent_sets(
        self,
        lp_solution: np.ndarray,
        node_list: List[int],
        graph: nx.Graph,
        dual_vars: np.ndarray,
    ) -> List[Set[int]]:
        """Extract profitable IS from LP solution using thresholding + pruning."""
        seen: set = set()
        profitable: List[Set[int]] = []

        def add_if_profitable(pruned: Set[int]) -> None:
            """Add to results if unique and profitable."""
            if not pruned:
                return
            # Local search refinement
            refined = _local_search(
                graph, pruned, dual_vars, max_passes=self.local_search_passes
            )
            sig = frozenset(refined)
            if sig not in seen:
                seen.add(sig)
                total = sum(dual_vars[v] for v in refined)
                if total > 1 + 1e-5:
                    profitable.append(set(refined))

        # Threshold-based extraction
        for threshold in self.support_thresholds:
            support = {
                node_list[i] for i in range(len(lp_solution))
                if lp_solution[i] > threshold
            }
            if not support:
                continue

            if self.multi_prune:
                add_if_profitable(_greedy_prune_dual_desc(support, graph, dual_vars))
                add_if_profitable(_greedy_prune_dual_asc(support, graph, dual_vars))
                for _ in range(self.num_random_prune_trials):
                    add_if_profitable(_greedy_prune_random(support, graph, self._rng))
            else:
                add_if_profitable(_greedy_prune_dual_desc(support, graph, dual_vars))

        # Randomized rounding
        if self.randomized_rounding:
            for _ in range(self.num_random_rounds):
                support = set()
                for i, p in enumerate(lp_solution):
                    if self._rng.random() < p:
                        support.add(node_list[i])
                if support:
                    add_if_profitable(_greedy_prune_dual_desc(support, graph, dual_vars))

        return profitable
