"""QCi Dirac-3 quantum annealing pricing oracle for MWIS."""

import random
import time
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from .base import PricingOracle
from ..timing import OracleTimer

try:
    import qci_client as qc
    from eqc_models.solvers import Dirac3ContinuousCloudSolver
    from eqc_models.base import QuadraticModel

    DIRAC_AVAILABLE = True
except ImportError:
    DIRAC_AVAILABLE = False


def _construct_gibbons_matrix(
    complement_graph: nx.Graph, weights: Dict[int, float]
) -> np.ndarray:
    """Build matrix B per Gibbons' Theorem 5 on the complement graph.

    B[i,i] = 1/w[i]
    B[i,j] = 0                         if (i,j) adjacent in complement
    B[i,j] = (1/w[i] + 1/w[j]) / 2    otherwise
    """
    nodes = list(complement_graph.nodes())
    n = len(nodes)
    B = np.zeros((n, n))
    for i, ni in enumerate(nodes):
        B[i, i] = 1.0 / weights[ni]
        for j, nj in enumerate(nodes):
            if i != j:
                if complement_graph.has_edge(ni, nj):
                    B[i, j] = 0.0
                else:
                    B[i, j] = (1.0 / weights[ni] + 1.0 / weights[nj]) / 2.0
    return B


def _dirac_solve_qp(
    adjacency_matrix: np.ndarray,
    num_samples: int = 100,
    relax_schedule: int = 2,
    sum_constraint: int = 1,
    solution_precision: Optional[float] = None,
) -> Optional[List[np.ndarray]]:
    """Submit a Motzkin-Straus QP to Dirac-3 and return all solution vectors."""
    n = adjacency_matrix.shape[0]
    if n == 0:
        return [np.array([])]

    C = np.zeros(n, dtype=np.float64)
    J = -0.5 * adjacency_matrix.astype(np.float64)

    model = QuadraticModel(C, J)
    model.upper_bound = np.ones(n, dtype=np.float64)
    solver = Dirac3ContinuousCloudSolver()

    params = {
        "sum_constraint": sum_constraint,
        "num_samples": num_samples,
        "relaxation_schedule": relax_schedule,
    }
    if solution_precision is not None:
        params["solution_precision"] = solution_precision

    response = solver.solve(model, **params)

    if response and "results" in response and "solutions" in response["results"]:
        solutions = response["results"]["solutions"]
        if solutions:
            return [np.array(s, dtype=np.float64) for s in solutions]
    return None


def _local_search(
    graph: nx.Graph,
    independent_set: Set[int],
    dual_vars: np.ndarray,
    max_passes: int = 5,
) -> Set[int]:
    """Improve an independent set via 1-swap local search.

    For each node not in the set with positive dual weight, check if swapping
    it in (removing its conflicting neighbors) yields a net gain.
    """
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


def _local_search_2swap(
    graph: nx.Graph,
    independent_set: Set[int],
    dual_vars: np.ndarray,
    max_passes: int = 3,
) -> Set[int]:
    """Improve an independent set via 2-swap local search.

    For each pair of IS nodes (u, v), try removing both and greedily adding
    from ALL positive-dual non-IS nodes (not just u/v neighbors). Accept if
    net dual weight increases. Restarts the pass on improvement to avoid
    stale iteration issues.

    More expensive than 1-swap (O(|IS|^2 * n) per pass) but can escape
    local optima that 1-swap misses, especially on denser graphs.
    """
    improved = set(independent_set)
    all_candidates = sorted(
        [v for v in graph.nodes() if dual_vars[v] > 1e-8],
        key=lambda v: dual_vars[v],
        reverse=True,
    )

    for _ in range(max_passes):
        found_improvement = False
        is_list = sorted(improved, key=lambda v: dual_vars[v])

        for i in range(len(is_list)):
            for j in range(i + 1, len(is_list)):
                u, v = is_list[i], is_list[j]
                removed_weight = dual_vars[u] + dual_vars[v]
                remaining = improved - {u, v}

                # Greedily add from ALL candidates not in remaining
                added = set()
                for w in all_candidates:
                    if w in remaining:
                        continue
                    if not any(graph.has_edge(w, a) for a in remaining | added):
                        added.add(w)

                added_weight = sum(dual_vars[w] for w in added)
                if added_weight > removed_weight + 1e-8:
                    improved = remaining | added
                    found_improvement = True
                    break  # restart pass with new IS
            if found_improvement:
                break  # restart pass with new IS

        if not found_improvement:
            break
    return improved


# ---------------------------------------------------------------------------
# Pruning strategies for multi-prune extraction
# ---------------------------------------------------------------------------

def _greedy_prune_dual_desc(
    support: Set[int],
    graph: nx.Graph,
    dual_vars: np.ndarray,
) -> Set[int]:
    """Greedy prune: highest dual weight first (baseline)."""
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
    """Greedy prune: lowest dual weight first (yields different IS)."""
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


# ---------------------------------------------------------------------------
# Solution vector clustering
# ---------------------------------------------------------------------------

def _cluster_solutions(
    solutions: List[np.ndarray],
    k: int = 5,
    max_iter: int = 50,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Cluster Dirac solution vectors via k-means.

    Returns:
        (centroids, extremes): Lists of k representative vectors each.
        centroids[i] = mean of cluster i (denoised consensus).
        extremes[i] = member farthest from centroid i (maximum diversity).
    """
    if len(solutions) <= k:
        return list(solutions), list(solutions)

    X = np.array(solutions)  # shape (num_samples, dim)
    n_samples, dim = X.shape

    # Simple k-means (avoid sklearn dependency)
    rng = np.random.RandomState(seed)
    indices = rng.choice(n_samples, size=k, replace=False)
    centers = X[indices].copy()

    for _ in range(max_iter):
        # Assign to nearest center
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        labels = dists.argmin(axis=1)

        # Update centers
        new_centers = np.zeros_like(centers)
        for c in range(k):
            members = X[labels == c]
            if len(members) > 0:
                new_centers[c] = members.mean(axis=0)
            else:
                new_centers[c] = centers[c]

        if np.allclose(centers, new_centers, atol=1e-8):
            break
        centers = new_centers

    # Collect centroids and extremes
    centroids = []
    extremes = []
    for c in range(k):
        mask = labels == c
        members = X[mask]
        if len(members) == 0:
            continue
        centroid = members.mean(axis=0)
        centroids.append(centroid)

        # Extreme = member farthest from centroid
        dists_to_center = np.linalg.norm(members - centroid, axis=1)
        extremes.append(members[dists_to_center.argmax()])

    return centroids, extremes


class DiracPricingOracle(PricingOracle):
    """Pricing oracle using QCi's Dirac-3 quantum annealing.

    Two approaches are supported:

    **Approach A — unweighted (default, ``method='filter'``)**
      Filter to the positive-dual subgraph, run unweighted MIS via Dirac on
      the complement graph's Motzkin-Straus QP, and check profitability.

    **Approach B — Gibbons weighted (``method='gibbons'``)**
      Build the Gibbons weighted Motzkin-Straus matrix with dual weights,
      submit the QP to Dirac, and extract the MWIS from the support.

    **Extraction enhancements** (see docs/refinement.md):
      - ``multi_prune=True``: Try multiple pruning orders (dual-desc, dual-asc,
        random) to extract more unique IS from each (solution, threshold) pair.
      - ``randomized_rounding=True``: Probabilistic node inclusion based on
        solution values, exploring more of the IS space.
    """

    def __init__(
        self,
        method: str = "gibbons",
        num_samples: int = 100,
        relax_schedule: int = 2,
        sum_constraint: int = 1,
        solution_precision: Optional[float] = None,
        support_threshold: Optional[float] = None,
        support_thresholds: Optional[List[float]] = None,
        local_search_passes: int = 5,
        multi_prune: bool = False,
        num_random_prune_trials: int = 3,
        randomized_rounding: bool = False,
        num_random_rounds: int = 10,
        random_seed: Optional[int] = None,
    ):
        if not DIRAC_AVAILABLE:
            raise ImportError(
                "Dirac oracle requires 'qci-client' and 'eqc-models' packages."
            )
        self.method = method
        self.num_samples = num_samples
        self.relax_schedule = relax_schedule
        self.sum_constraint = sum_constraint
        self.solution_precision = solution_precision
        self.local_search_passes = local_search_passes

        # Multi-prune: use multiple pruning strategies per (solution, threshold)
        self.multi_prune = multi_prune
        self.num_random_prune_trials = num_random_prune_trials

        # Randomized rounding: probabilistic extraction
        self.randomized_rounding = randomized_rounding
        self.num_random_rounds = num_random_rounds

        # RNG for reproducibility
        self._rng = random.Random(random_seed)

        self.timer = OracleTimer()

        # Support backward compatibility: singular support_threshold -> list
        if support_thresholds is not None:
            self.support_thresholds = support_thresholds
        elif support_threshold is not None:
            self.support_thresholds = [support_threshold]
        else:
            self.support_thresholds = [0.005, 0.01, 0.05, 0.1, 0.2]

    # ------------------------------------------------------------------
    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        if self.method == "filter":
            return self._solve_filter(graph, dual_vars)
        return self._solve_gibbons(graph, dual_vars)

    # ------------------------------------------------------------------
    def _extract_independent_sets(
        self,
        solutions: List[np.ndarray],
        node_list: List[int],
        graph: nx.Graph,
        dual_vars: np.ndarray,
    ) -> List[Set[int]]:
        """Extract profitable IS from multiple Dirac solution vectors.

        Tries all (solution, threshold) combinations, applies pruning strategies
        and local search, deduplicates, and returns all profitable sets.

        With multi_prune=True, tries multiple pruning orders per support set.
        With randomized_rounding=True, also uses probabilistic extraction.
        """
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

        # Standard threshold-based extraction
        for x in solutions:
            for threshold in self.support_thresholds:
                support = {
                    node_list[i] for i in range(len(x)) if x[i] > threshold
                }
                if not support:
                    continue

                if self.multi_prune:
                    # Try multiple pruning strategies
                    add_if_profitable(_greedy_prune_dual_desc(support, graph, dual_vars))
                    add_if_profitable(_greedy_prune_dual_asc(support, graph, dual_vars))
                    for _ in range(self.num_random_prune_trials):
                        add_if_profitable(_greedy_prune_random(support, graph, self._rng))
                else:
                    # Baseline: only dual-descending prune
                    add_if_profitable(_greedy_prune_dual_desc(support, graph, dual_vars))

        # Randomized rounding extraction
        if self.randomized_rounding:
            for x in solutions:
                x_sum = x.sum()
                if x_sum < 1e-10:
                    continue
                x_norm = x / x_sum  # normalize to probabilities

                for _ in range(self.num_random_rounds):
                    # Sample nodes proportionally to solution values (scaled up)
                    support = set()
                    for i, p in enumerate(x_norm):
                        if self._rng.random() < p * 3:  # scale factor
                            support.add(node_list[i])

                    if support:
                        add_if_profitable(
                            _greedy_prune_dual_desc(support, graph, dual_vars)
                        )

        return profitable

    # ------------------------------------------------------------------
    # Approach A: unweighted MIS on positive-dual subgraph
    # ------------------------------------------------------------------
    def _solve_filter(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        pos_nodes = [v for v in graph.nodes() if dual_vars[v] > 1e-5]
        if not pos_nodes:
            return []

        subgraph = graph.subgraph(pos_nodes)
        if not subgraph.edges():
            total = sum(dual_vars[v] for v in pos_nodes)
            return [set(pos_nodes)] if total > 1 + 1e-5 else []

        complement = nx.complement(subgraph)
        adj = nx.to_numpy_array(complement, nodelist=sorted(subgraph.nodes()))

        t0 = time.monotonic()
        solutions = _dirac_solve_qp(
            adj,
            num_samples=self.num_samples,
            relax_schedule=self.relax_schedule,
            sum_constraint=self.sum_constraint,
            solution_precision=self.solution_precision,
        )
        api_seconds = time.monotonic() - t0

        if solutions is None:
            self.timer.record(api_seconds=api_seconds)
            return []

        node_list = sorted(subgraph.nodes())
        t0 = time.monotonic()
        result = self._extract_independent_sets(solutions, node_list, graph, dual_vars)
        extract_seconds = time.monotonic() - t0

        self.timer.record(
            api_seconds=api_seconds,
            extract_seconds=extract_seconds,
            columns_found=len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Approach B: weighted Gibbons matrix -> Dirac -> MWIS
    # ------------------------------------------------------------------
    def _solve_gibbons(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        pos_nodes = [v for v in graph.nodes() if dual_vars[v] > 1e-5]
        if not pos_nodes:
            return []

        subgraph = graph.subgraph(pos_nodes)
        if not subgraph.edges():
            total = sum(dual_vars[v] for v in pos_nodes)
            return [set(pos_nodes)] if total > 1 + 1e-5 else []

        weights = {v: float(dual_vars[v]) for v in pos_nodes}
        complement = nx.complement(subgraph)
        B = _construct_gibbons_matrix(complement, weights)

        # Dirac minimises  x^T B x  on the simplex.
        # The model format is  min  x^T C + x^T J x, so C=0, J=B
        n = B.shape[0]
        C = np.zeros(n, dtype=np.float64)
        J = B.astype(np.float64)

        model = QuadraticModel(C, J)
        model.upper_bound = np.ones(n, dtype=np.float64)
        solver = Dirac3ContinuousCloudSolver()

        params = {
            "sum_constraint": self.sum_constraint,
            "num_samples": self.num_samples,
            "relaxation_schedule": self.relax_schedule,
        }
        if self.solution_precision is not None:
            params["solution_precision"] = self.solution_precision

        t0 = time.monotonic()
        response = solver.solve(model, **params)
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

        t0 = time.monotonic()
        result = self._extract_independent_sets(solutions, node_list, graph, dual_vars)
        extract_seconds = time.monotonic() - t0

        self.timer.record(
            api_seconds=api_seconds,
            extract_seconds=extract_seconds,
            columns_found=len(result),
        )
        return result
