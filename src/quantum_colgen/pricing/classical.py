"""Classical MILP-based pricing subproblem solver."""

from typing import List, Set

import networkx as nx
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

from .base import PricingOracle


class ClassicalPricingOracle(PricingOracle):
    """Exact MILP solver for the Maximum Weight Independent Set pricing subproblem.

    Formulation (on the positive-dual subgraph V'):
        maximize  sum  dual_vars[v] * x[v]   for v in V'
        s.t.      x[u] + x[v] <= 1           for all edges (u,v) in G[V']
                  x[v] in {0, 1}
    """

    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        node_list = sorted(graph.nodes())
        num_vertices = len(node_list)

        if num_vertices == 0:
            return []

        # Filter to positive-dual subgraph V' = {v | dual_vars[v] > 0}
        pos_indices = [i for i in range(num_vertices) if dual_vars[i] > 1e-10]
        if not pos_indices:
            return []

        filtered_nodes = [node_list[i] for i in pos_indices]
        filtered_weights = dual_vars[pos_indices]
        filtered_node_to_idx = {node: i for i, node in enumerate(filtered_nodes)}
        subgraph = graph.subgraph(filtered_nodes)

        n_filt = len(filtered_nodes)

        # Objective: maximize weighted sum -> minimize negative
        c_psp = -filtered_weights

        # Edge constraints
        edges = list(subgraph.edges())
        if edges:
            A_ub = lil_matrix((len(edges), n_filt), dtype=float)
            for i, (u, v) in enumerate(edges):
                A_ub[i, filtered_node_to_idx[u]] = 1
                A_ub[i, filtered_node_to_idx[v]] = 1
            b_ub = np.ones(len(edges))
            constraints = [LinearConstraint(A_ub.toarray(), -np.inf, b_ub)]
        else:
            constraints = []

        integrality = np.ones(n_filt, dtype=int)
        result = milp(
            c=c_psp,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb=0, ub=1),
        )

        if not result.success:
            return []

        selected = [filtered_nodes[j] for j in range(n_filt) if result.x[j] > 0.5]
        if not selected:
            return []

        total_weight = sum(dual_vars[v] for v in selected)
        if total_weight > 1.0 + 1e-6:
            return [set(selected)]
        return []
