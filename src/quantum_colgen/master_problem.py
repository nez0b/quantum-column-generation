"""Restricted Master Problem (RMP) solver and final ILP for graph coloring."""

from typing import FrozenSet, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

# Optional Hexaly import with graceful fallback
try:
    import hexaly.optimizer as hx
    HEXALY_AVAILABLE = True
except ImportError:
    HEXALY_AVAILABLE = False


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


def _solve_final_ilp_highs(
    columns: List[FrozenSet[int]],
    num_vertices: int,
    time_limit: Optional[int] = None,
) -> Tuple[Optional[int], List[int]]:
    """Solve final ILP using HiGHS (via scipy.optimize.milp)."""
    num_columns = len(columns)

    A = np.zeros((num_vertices, num_columns))
    for s_idx, col in enumerate(columns):
        for v in col:
            A[v, s_idx] = 1

    c = np.ones(num_columns)
    b = np.ones(num_vertices)
    constraints = [LinearConstraint(A, b, b)]
    integrality = np.ones(num_columns, dtype=int)

    options = {}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=Bounds(lb=0, ub=1),
        options=options if options else None,
    )
    if not result.success:
        return None, []

    selected = [i for i, val in enumerate(result.x) if val > 0.5]
    return int(round(result.fun)), selected


def _solve_final_ilp_hexaly(
    columns: List[FrozenSet[int]],
    num_vertices: int,
    time_limit: Optional[int] = None,
) -> Tuple[Optional[int], List[int]]:
    """Solve final ILP using Hexaly optimizer.

    minimize   sum y_s
    s.t.       sum_{s : v in s} y_s  = 1    for all v
               y_s in {0, 1}
    """
    if not HEXALY_AVAILABLE:
        raise RuntimeError("Hexaly is not installed. Install with: uv pip install hexaly")

    num_columns = len(columns)
    if num_columns == 0:
        return None, []

    # Build column-to-vertices mapping for efficient constraint building
    col_to_vertices = [list(col) for col in columns]

    # Build vertex-to-columns mapping (which columns cover each vertex)
    vertex_to_cols: List[List[int]] = [[] for _ in range(num_vertices)]
    for s_idx, col in enumerate(columns):
        for v in col:
            vertex_to_cols[v].append(s_idx)

    with hx.HexalyOptimizer() as optimizer:
        model = optimizer.model

        # Binary variables: one per column
        y = [model.bool() for _ in range(num_columns)]

        # Coverage constraints: each vertex must be covered exactly once
        for v in range(num_vertices):
            covering_cols = [y[s] for s in vertex_to_cols[v]]
            if covering_cols:
                model.constraint(model.sum(covering_cols) == 1)
            else:
                # Vertex not covered by any column - problem is infeasible
                return None, []

        # Objective: minimize total number of colors
        model.minimize(model.sum(y))

        model.close()

        # Set time limit
        if time_limit is not None:
            optimizer.param.time_limit = time_limit

        # Suppress output
        optimizer.param.verbosity = 0

        optimizer.solve()

        # Check solution status
        solution = optimizer.solution
        status = solution.status
        # Status: FEASIBLE or OPTIMAL means we have a solution
        if status == hx.HxSolutionStatus.INFEASIBLE:
            return None, []
        if status == hx.HxSolutionStatus.INCONSISTENT:
            return None, []

        # Extract solution
        selected = [i for i in range(num_columns) if solution.get_value(y[i]) == 1]
        num_colors = len(selected)

        return num_colors, selected


def solve_final_ilp(
    columns: List[FrozenSet[int]],
    num_vertices: int,
    solver: str = "highs",
    time_limit: Optional[int] = None,
) -> Tuple[Optional[int], List[int]]:
    """Solve the final integer program to obtain an optimal coloring.

    minimize   sum y_s
    s.t.       sum_{s : v in s} y_s  = 1    for all v
               y_s in {0, 1}

    Args:
        columns: Full set of generated columns.
        num_vertices: Number of graph vertices.
        solver: ILP solver to use ("highs" or "hexaly").
        time_limit: Optional time limit in seconds. If reached, returns best
            feasible solution found (may not be optimal).

    Returns:
        (num_colors, selected_column_indices) or (None, []) on failure.
    """
    if solver == "hexaly":
        return _solve_final_ilp_hexaly(columns, num_vertices, time_limit)
    elif solver == "highs":
        return _solve_final_ilp_highs(columns, num_vertices, time_limit)
    else:
        raise ValueError(f"Unknown ILP solver: {solver}. Use 'highs' or 'hexaly'.")
