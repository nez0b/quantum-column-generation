"""Abstract base class for pricing oracles."""

from abc import ABC, abstractmethod
from typing import List, Set

import networkx as nx
import numpy as np


class PricingOracle(ABC):
    """Interface for column-generation pricing subproblem solvers.

    A pricing oracle finds independent sets whose total dual weight exceeds 1
    (i.e. negative reduced cost), which can be added as new columns to the
    restricted master problem.
    """

    @abstractmethod
    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        """Return profitable independent sets.

        An independent set S is *profitable* when sum(dual_vars[v] for v in S) > 1.

        Args:
            graph: The original graph (nodes labelled 0 .. n-1).
            dual_vars: Dual variables from the RMP, one per vertex.

        Returns:
            A list of independent sets (each a set of node indices) that have
            negative reduced cost.  May be empty when no profitable column exists.
        """
