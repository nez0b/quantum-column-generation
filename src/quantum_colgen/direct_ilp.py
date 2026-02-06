"""Direct ILP formulation for minimum vertex graph coloring.

This module provides exact ILP solvers for graph coloring, separate from the
column generation approach. Useful for comparison on small graphs (<50 nodes).

ILP Formulation:
- x[i,k] ∈ {0,1}: node i is assigned color k
- y[k] ∈ {0,1}: color k is used
- Minimize: Σ_k y[k]
- Subject to:
  - Σ_k x[i,k] = 1 for all nodes i (each node gets one color)
  - x[i,k] + x[j,k] ≤ 1 for all edges (i,j) and colors k (conflict)
  - x[i,k] ≤ y[k] for all i,k (linking)
"""

import time
from typing import Optional, Tuple, List, Set

import networkx as nx
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def greedy_chromatic_upper_bound(graph: nx.Graph) -> int:
    """Get upper bound on chromatic number using greedy coloring."""
    coloring = nx.coloring.greedy_color(graph, strategy="largest_first")
    return max(coloring.values()) + 1 if coloring else 1


def solve_coloring_ilp_highs(
    graph: nx.Graph,
    max_colors: Optional[int] = None,
    time_limit: Optional[float] = None,
) -> Tuple[Optional[int], Optional[List[Set[int]]], float, dict]:
    """Solve minimum vertex coloring using scipy MILP (HiGHS).

    Args:
        graph: NetworkX graph to color.
        max_colors: Upper bound on number of colors. If None, uses greedy result.
        time_limit: Time limit in seconds (not directly supported by scipy,
            but we track it and return partial results).

    Returns:
        Tuple of (chi, color_classes, solve_time, info_dict).
        chi is None if no solution found.
    """
    start_time = time.time()

    n = graph.number_of_nodes()
    if n == 0:
        return 0, [], 0.0, {"status": "trivial"}

    nodes = list(graph.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    edges = list(graph.edges())
    m = len(edges)

    # Upper bound on colors
    if max_colors is None:
        max_colors = greedy_chromatic_upper_bound(graph)
    K = max_colors

    # Variables: x[i,k] for i=0..n-1, k=0..K-1, then y[k] for k=0..K-1
    # Layout: x[0,0], x[0,1], ..., x[0,K-1], x[1,0], ..., x[n-1,K-1], y[0], ..., y[K-1]
    num_x = n * K
    num_y = K
    num_vars = num_x + num_y

    def x_idx(i, k):
        return i * K + k

    def y_idx(k):
        return num_x + k

    # Objective: minimize Σ_k y[k]
    c = np.zeros(num_vars)
    for k in range(K):
        c[y_idx(k)] = 1.0

    # All variables are binary
    integrality = np.ones(num_vars, dtype=int)
    bounds = Bounds(lb=np.zeros(num_vars), ub=np.ones(num_vars))

    # Build constraints
    # 1. Assignment: Σ_k x[i,k] = 1 for all i (n equality constraints)
    # 2. Conflict: x[i,k] + x[j,k] ≤ 1 for all edges and colors (m*K constraints)
    # 3. Linking: x[i,k] ≤ y[k], i.e., x[i,k] - y[k] ≤ 0 for all i,k (n*K constraints)

    num_constraints = n + m * K + n * K
    A = np.zeros((num_constraints, num_vars))
    lb = np.zeros(num_constraints)
    ub = np.zeros(num_constraints)

    row = 0

    # Assignment constraints (equality)
    for i in range(n):
        for k in range(K):
            A[row, x_idx(i, k)] = 1.0
        lb[row] = 1.0
        ub[row] = 1.0
        row += 1

    # Conflict constraints (edges)
    for (u, v) in edges:
        i = node_to_idx[u]
        j = node_to_idx[v]
        for k in range(K):
            A[row, x_idx(i, k)] = 1.0
            A[row, x_idx(j, k)] = 1.0
            lb[row] = -np.inf
            ub[row] = 1.0
            row += 1

    # Linking constraints
    for i in range(n):
        for k in range(K):
            A[row, x_idx(i, k)] = 1.0
            A[row, y_idx(k)] = -1.0
            lb[row] = -np.inf
            ub[row] = 0.0
            row += 1

    constraints = LinearConstraint(A, lb, ub)

    # Solve
    options = {}
    if time_limit is not None:
        options["time_limit"] = time_limit

    try:
        result = milp(c, integrality=integrality, bounds=bounds,
                      constraints=constraints, options=options)
    except Exception as e:
        solve_time = time.time() - start_time
        return None, None, solve_time, {"status": "error", "message": str(e)}

    solve_time = time.time() - start_time

    # Check if we have a solution (even if not optimal/timed out)
    has_solution = result.x is not None and len(result.x) == num_vars

    if not has_solution:
        return None, None, solve_time, {
            "status": result.message,
            "n": n,
            "m": m,
            "K": K,
            "num_vars": num_vars,
            "num_constraints": num_constraints,
        }

    # Extract solution (may be feasible but not proven optimal)
    x_vals = result.x
    chi = int(round(result.fun))

    # Build color classes
    color_classes = [set() for _ in range(K)]
    for i in range(n):
        for k in range(K):
            if x_vals[x_idx(i, k)] > 0.5:
                color_classes[k].add(nodes[i])
                break

    # Filter empty color classes
    color_classes = [cc for cc in color_classes if len(cc) > 0]

    # Determine status
    if result.success:
        status = "optimal"
    elif "time limit" in result.message.lower():
        status = "feasible (time limit)"
    else:
        status = f"feasible ({result.message})"

    info = {
        "status": status,
        "n": n,
        "m": m,
        "K": K,
        "num_vars": num_vars,
        "num_constraints": num_constraints,
        "solver": "HiGHS",
        "mip_gap": getattr(result, 'mip_gap', None),
    }

    return chi, color_classes, solve_time, info


def solve_coloring_ilp_hexaly(
    graph: nx.Graph,
    max_colors: Optional[int] = None,
    time_limit: Optional[float] = 60.0,
) -> Tuple[Optional[int], Optional[List[Set[int]]], float, dict]:
    """Solve minimum vertex coloring using Hexaly.

    Args:
        graph: NetworkX graph to color.
        max_colors: Upper bound on number of colors. If None, uses greedy result.
        time_limit: Time limit in seconds. Default 60s.

    Returns:
        Tuple of (chi, color_classes, solve_time, info_dict).
        chi is None if no solution found.
    """
    start_time = time.time()

    try:
        import hexaly.optimizer as hx
    except ImportError:
        solve_time = time.time() - start_time
        return None, None, solve_time, {
            "status": "error",
            "message": "Hexaly not installed. Set PYTHONPATH and DYLD_LIBRARY_PATH.",
        }

    n = graph.number_of_nodes()
    if n == 0:
        return 0, [], 0.0, {"status": "trivial"}

    nodes = list(graph.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    edges = list(graph.edges())
    m = len(edges)

    # Upper bound on colors
    if max_colors is None:
        max_colors = greedy_chromatic_upper_bound(graph)
    K = max_colors

    with hx.HexalyOptimizer() as optimizer:
        model = optimizer.model

        # Decision variables
        # x[i,k] = 1 if node i has color k
        x = [[model.bool() for k in range(K)] for i in range(n)]
        # y[k] = 1 if color k is used
        y = [model.bool() for k in range(K)]

        # Constraints
        # 1. Each node gets exactly one color
        for i in range(n):
            model.constraint(model.sum(x[i][k] for k in range(K)) == 1)

        # 2. Adjacent nodes have different colors
        for (u, v) in edges:
            i = node_to_idx[u]
            j = node_to_idx[v]
            for k in range(K):
                model.constraint(x[i][k] + x[j][k] <= 1)

        # 3. Linking: if node i has color k, then color k is used
        for i in range(n):
            for k in range(K):
                model.constraint(x[i][k] <= y[k])

        # Objective: minimize number of colors used
        num_colors = model.sum(y[k] for k in range(K))
        model.minimize(num_colors)

        model.close()

        # Set time limit
        if time_limit is not None:
            optimizer.param.time_limit = int(time_limit)

        optimizer.solve()

        solve_time = time.time() - start_time

        # Check solution status
        status = optimizer.solution.status
        if status == hx.HxSolutionStatus.INFEASIBLE:
            return None, None, solve_time, {
                "status": "infeasible",
                "n": n,
                "m": m,
                "K": K,
            }

        if status == hx.HxSolutionStatus.FEASIBLE:
            sol_status = "feasible"
        elif status == hx.HxSolutionStatus.OPTIMAL:
            sol_status = "optimal"
        else:
            sol_status = str(status)

        # Extract solution
        chi = int(optimizer.solution.get_value(num_colors))

        # Build color classes
        color_classes = [set() for _ in range(K)]
        for i in range(n):
            for k in range(K):
                if optimizer.solution.get_value(x[i][k]) == 1:
                    color_classes[k].add(nodes[i])
                    break

        # Filter empty color classes
        color_classes = [cc for cc in color_classes if len(cc) > 0]

        # Get bound info
        gap = None
        lb = None
        try:
            # Try different API methods for getting bounds
            if hasattr(optimizer, 'statistics'):
                stats = optimizer.statistics
                if hasattr(stats, 'get_lower_bound'):
                    lb = stats.get_lower_bound()
            # Compute gap from status if optimal
            if sol_status == "optimal":
                gap = 0.0
                lb = chi
        except Exception:
            pass

        info = {
            "status": sol_status,
            "n": n,
            "m": m,
            "K": K,
            "num_vars": n * K + K,
            "num_constraints": n + m * K + n * K,
            "solver": "Hexaly",
            "lower_bound": lb,
            "gap": gap,
        }

        return chi, color_classes, solve_time, info


def validate_coloring(graph: nx.Graph, color_classes: List[Set[int]]) -> bool:
    """Validate that a coloring is valid (no adjacent nodes share color)."""
    nodes_covered = set()
    for cc in color_classes:
        # Check independence
        subgraph = graph.subgraph(cc)
        if subgraph.number_of_edges() > 0:
            return False
        nodes_covered.update(cc)

    # Check all nodes covered
    return nodes_covered == set(graph.nodes())
