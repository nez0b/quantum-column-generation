"""Restricted Master Problem (RMP) solver and final ILP for graph coloring."""

from typing import FrozenSet, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def solve_rmp(
    columns: List[FrozenSet[int]],
    num_vertices: int,
) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Solve the LP relaxation of the Restricted Master Problem.

    minimize   sum y_s
    s.t.       sum_{s : v in s} y_s  = 1    for all v
               0 <= y_s <= 1

    Args:
        columns: Current set of independent-set columns (each a frozenset of node indices).
        num_vertices: Total number of vertices in the graph.

    Returns:
        (objective_value, dual_variables) or (None, None) on failure.
    """
    num_columns = len(columns)

    A = lil_matrix((num_vertices, num_columns))
    for s_idx, col in enumerate(columns):
        for v in col:
            A[v, s_idx] = 1
    A = A.tocsc()

    b = np.ones(num_vertices)
    c = np.ones(num_columns)
    bounds = [(0, 1)] * num_columns

    result = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")
    if not result.success:
        return None, None

    # Extract dual variables
    if hasattr(result, "eqlin") and result.eqlin is not None:
        duals = (
            result.eqlin["marginals"]
            if isinstance(result.eqlin, dict)
            else result.eqlin.marginals
        )
        if duals is not None:
            return result.fun, np.asarray(duals, dtype=float)
    return result.fun, np.zeros(num_vertices)


def solve_final_ilp(
    columns: List[FrozenSet[int]],
    num_vertices: int,
) -> Tuple[Optional[int], List[int]]:
    """Solve the final integer program to obtain an optimal coloring.

    minimize   sum y_s
    s.t.       sum_{s : v in s} y_s  = 1    for all v
               y_s in {0, 1}

    Args:
        columns: Full set of generated columns.
        num_vertices: Number of graph vertices.

    Returns:
        (num_colors, selected_column_indices) or (None, []) on failure.
    """
    num_columns = len(columns)

    A = np.zeros((num_vertices, num_columns))
    for s_idx, col in enumerate(columns):
        for v in col:
            A[v, s_idx] = 1

    c = np.ones(num_columns)
    b = np.ones(num_vertices)
    constraints = [LinearConstraint(A, b, b)]
    integrality = np.ones(num_columns, dtype=int)

    result = milp(c=c, constraints=constraints, integrality=integrality, bounds=Bounds(lb=0, ub=1))
    if not result.success:
        return None, []

    selected = [i for i, val in enumerate(result.x) if val > 0.5]
    return int(round(result.fun)), selected
