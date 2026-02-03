"""Column generation loop orchestrator for minimum vertex graph coloring."""

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Any

import networkx as nx
import numpy as np

from .master_problem import solve_rmp, solve_final_ilp
from .pricing.base import PricingOracle


def column_generation(
    graph: nx.Graph,
    oracle: PricingOracle,
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    verbose: bool = False,
    ilp_solver: str = "highs",
    ilp_time_limit: Optional[int] = None,
) -> Tuple[Optional[int], List[FrozenSet[int]], Dict[str, Any]]:
    """Run column generation for minimum vertex coloring.

    Args:
        graph: Input graph (nodes 0..n-1).
        oracle: Pricing oracle that finds profitable independent sets.
        max_iterations: Maximum CG iterations.
        tolerance: Convergence tolerance (unused directly; profitability
            is checked inside the oracle).
        verbose: Print iteration details.
        ilp_solver: Solver for final ILP ("highs" or "hexaly").
        ilp_time_limit: Optional time limit for final ILP in seconds.

    Returns:
        (chromatic_number, coloring_as_frozensets, stats_dict)
        chromatic_number is None on failure.
    """
    node_list = sorted(graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    num_vertices = len(node_list)

    # Initialise with singleton columns
    columns: List[FrozenSet[int]] = [frozenset([i]) for i in range(num_vertices)]
    known_sigs = {tuple(sorted(c)) for c in columns}

    stats: Dict[str, Any] = {"iterations": 0, "columns_generated": 0}

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"--- Iteration {iteration} ---")

        # Solve RMP
        obj, dual_vars = solve_rmp(columns, num_vertices)
        if dual_vars is None:
            if verbose:
                print("  RMP failed")
            break

        if verbose:
            print(f"  RMP obj={obj:.4f}  duals={dual_vars}")

        # Pricing subproblem
        new_cols = oracle.solve(graph, dual_vars)

        if not new_cols:
            if verbose:
                print("  No profitable columns — converged")
            break

        added = 0
        for col_set in new_cols:
            sig = tuple(sorted(col_set))
            if sig not in known_sigs:
                columns.append(frozenset(col_set))
                known_sigs.add(sig)
                added += 1

        if verbose:
            print(f"  Added {added} columns, total {len(columns)}")

        if added == 0:
            if verbose:
                print("  All columns already known — converged")
            break

        stats["iterations"] = iteration

    stats["columns_generated"] = len(columns)

    # Final ILP
    num_colors, selected_indices = solve_final_ilp(
        columns, num_vertices, solver=ilp_solver, time_limit=ilp_time_limit
    )
    if num_colors is None:
        return None, [], stats

    coloring = [columns[i] for i in selected_indices]
    stats["num_colors"] = num_colors
    return num_colors, coloring, stats


def verify_coloring(graph: nx.Graph, coloring: List[FrozenSet[int]]) -> bool:
    """Check that *coloring* is a valid proper coloring of *graph*.

    Returns True iff:
      1. Every vertex appears in exactly one color class.
      2. Each color class is an independent set.
    """
    all_vertices = set(graph.nodes())
    covered: Set[int] = set()
    for color_set in coloring:
        for v in color_set:
            if v in covered:
                return False  # vertex in two color classes
            covered.add(v)
        for u in color_set:
            for w in color_set:
                if u != w and graph.has_edge(u, w):
                    return False
    return covered == all_vertices


@dataclass
class ValidationResult:
    """Detailed coloring validation result."""
    valid: bool
    num_colors: int
    num_vertices_covered: int
    num_vertices_expected: int
    missing_vertices: List[int]
    duplicate_vertices: List[int]
    edge_violations: List[Tuple[int, int, int]]  # (color_idx, u, v)
    empty_color_classes: List[int]


def validate_coloring(
    graph: nx.Graph, coloring: List[FrozenSet[int]],
) -> ValidationResult:
    """Detailed validation of a coloring solution.

    Returns a ValidationResult with specific error information.
    """
    all_vertices = set(graph.nodes())
    covered: Set[int] = set()
    duplicates: List[int] = []
    edge_violations: List[Tuple[int, int, int]] = []
    empty_classes: List[int] = []

    for idx, color_set in enumerate(coloring):
        if len(color_set) == 0:
            empty_classes.append(idx)
            continue
        for v in color_set:
            if v in covered:
                duplicates.append(v)
            covered.add(v)
        # Check independence
        nodes = sorted(color_set)
        for i, u in enumerate(nodes):
            for w in nodes[i + 1:]:
                if graph.has_edge(u, w):
                    edge_violations.append((idx, u, w))

    missing = sorted(all_vertices - covered)
    valid = (
        len(missing) == 0
        and len(duplicates) == 0
        and len(edge_violations) == 0
    )

    return ValidationResult(
        valid=valid,
        num_colors=len(coloring),
        num_vertices_covered=len(covered),
        num_vertices_expected=len(all_vertices),
        missing_vertices=missing,
        duplicate_vertices=duplicates,
        edge_violations=edge_violations,
        empty_color_classes=empty_classes,
    )
