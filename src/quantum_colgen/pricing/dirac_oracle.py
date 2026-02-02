"""QCi Dirac-3 quantum annealing pricing oracle for MWIS."""

from typing import Dict, List, Optional, Set

import networkx as nx
import numpy as np

from .base import PricingOracle

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
) -> Optional[np.ndarray]:
    """Submit a Motzkin-Straus QP to Dirac-3 and return best solution vector."""
    n = adjacency_matrix.shape[0]
    if n == 0:
        return np.array([])

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
            return np.array(solutions[0], dtype=np.float64)
    return None


class DiracPricingOracle(PricingOracle):
    """Pricing oracle using QCi's Dirac-3 quantum annealing.

    Two approaches are supported:

    **Approach A — unweighted (default, ``method='filter'``)**
      Filter to the positive-dual subgraph, run unweighted MIS via Dirac on
      the complement graph's Motzkin-Straus QP, and check profitability.

    **Approach B — Gibbons weighted (``method='gibbons'``)**
      Build the Gibbons weighted Motzkin-Straus matrix with dual weights,
      submit the QP to Dirac, and extract the MWIS from the support.
    """

    def __init__(
        self,
        method: str = "gibbons",
        num_samples: int = 100,
        relax_schedule: int = 2,
        sum_constraint: int = 1,
        solution_precision: Optional[float] = None,
        support_threshold: float = 0.01,
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
        self.support_threshold = support_threshold

    # ------------------------------------------------------------------
    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        if self.method == "filter":
            return self._solve_filter(graph, dual_vars)
        return self._solve_gibbons(graph, dual_vars)

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

        x = _dirac_solve_qp(
            adj,
            num_samples=self.num_samples,
            relax_schedule=self.relax_schedule,
            sum_constraint=self.sum_constraint,
            solution_precision=self.solution_precision,
        )
        if x is None:
            return []

        node_list = sorted(subgraph.nodes())
        support = {node_list[i] for i in range(len(x)) if x[i] > self.support_threshold}
        if not support:
            return []

        # Greedy prune to ensure independence (highest dual weight first)
        pruned: Set[int] = set()
        for v in sorted(support, key=lambda v: dual_vars[v], reverse=True):
            if not any(graph.has_edge(v, u) for u in pruned):
                pruned.add(v)

        total = sum(dual_vars[v] for v in pruned)
        return [pruned] if total > 1 + 1e-5 else []

    # ------------------------------------------------------------------
    # Approach B: weighted Gibbons matrix → Dirac → MWIS
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

        response = solver.solve(model, **params)
        if not (response and "results" in response and "solutions" in response["results"]):
            return []

        solutions = response["results"]["solutions"]
        if not solutions:
            return []

        x = np.array(solutions[0], dtype=np.float64)
        node_list = sorted(subgraph.nodes())
        support = {node_list[i] for i in range(len(x)) if x[i] > self.support_threshold}
        if not support:
            return []

        # Greedy prune to ensure independence
        pruned: Set[int] = set()
        for v in sorted(support, key=lambda v: dual_vars[v], reverse=True):
            if not any(graph.has_edge(v, u) for u in pruned):
                pruned.add(v)

        total = sum(dual_vars[v] for v in pruned)
        return [pruned] if total > 1 + 1e-5 else []
