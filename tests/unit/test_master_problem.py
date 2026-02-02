"""Tests for the RMP LP relaxation and final ILP."""

import numpy as np
import pytest

from quantum_colgen.master_problem import solve_rmp, solve_final_ilp


class TestSolveRMP:
    def test_singleton_columns_triangle(self, graph_triangle):
        """Singletons on K3 should give LP obj = 3 (no IS larger than 1)."""
        columns = [frozenset([i]) for i in range(3)]
        obj, duals = solve_rmp(columns, 3)
        assert obj is not None
        assert abs(obj - 3.0) < 1e-4
        assert duals is not None and len(duals) == 3

    def test_optimal_columns_path(self, graph_p4):
        """P4 is 2-colourable: {0,2} and {1,3}."""
        columns = [
            frozenset([0]),
            frozenset([1]),
            frozenset([2]),
            frozenset([3]),
            frozenset([0, 2]),
            frozenset([1, 3]),
        ]
        obj, duals = solve_rmp(columns, 4)
        assert obj is not None
        assert obj <= 2.0 + 1e-4

    def test_duals_nonnegative(self, graph_5vertex):
        columns = [frozenset([i]) for i in range(5)]
        _, duals = solve_rmp(columns, 5)
        assert duals is not None
        assert all(d >= -1e-8 for d in duals)


class TestSolveFinalILP:
    def test_triangle_needs_3_colors(self, graph_triangle):
        columns = [frozenset([0]), frozenset([1]), frozenset([2])]
        num_colors, selected = solve_final_ilp(columns, 3)
        assert num_colors == 3

    def test_path_2_colors(self, graph_p4):
        columns = [
            frozenset([0]),
            frozenset([1]),
            frozenset([2]),
            frozenset([3]),
            frozenset([0, 2]),
            frozenset([1, 3]),
        ]
        num_colors, selected = solve_final_ilp(columns, 4)
        assert num_colors == 2
