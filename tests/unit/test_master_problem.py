"""Tests for the RMP LP relaxation and final ILP."""

import numpy as np
import pytest

from quantum_colgen.master_problem import solve_rmp, solve_final_ilp, HEXALY_AVAILABLE


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

    def test_highs_with_time_limit(self, graph_p4):
        """HiGHS should work with time limit parameter."""
        columns = [
            frozenset([0]),
            frozenset([1]),
            frozenset([2]),
            frozenset([3]),
            frozenset([0, 2]),
            frozenset([1, 3]),
        ]
        num_colors, selected = solve_final_ilp(columns, 4, solver="highs", time_limit=60)
        assert num_colors == 2

    def test_invalid_solver_raises(self):
        """Unknown solver should raise ValueError."""
        columns = [frozenset([0]), frozenset([1])]
        with pytest.raises(ValueError, match="Unknown ILP solver"):
            solve_final_ilp(columns, 2, solver="invalid")


@pytest.mark.skipif(not HEXALY_AVAILABLE, reason="Hexaly not installed")
class TestHexalySolver:
    """Tests for Hexaly ILP solver."""

    def test_hexaly_triangle_needs_3_colors(self, graph_triangle):
        """K3 requires 3 colors with Hexaly."""
        columns = [frozenset([0]), frozenset([1]), frozenset([2])]
        num_colors, selected = solve_final_ilp(columns, 3, solver="hexaly")
        assert num_colors == 3
        assert len(selected) == 3

    def test_hexaly_path_2_colors(self, graph_p4):
        """P4 is 2-colorable with Hexaly."""
        columns = [
            frozenset([0]),
            frozenset([1]),
            frozenset([2]),
            frozenset([3]),
            frozenset([0, 2]),
            frozenset([1, 3]),
        ]
        num_colors, selected = solve_final_ilp(columns, 4, solver="hexaly")
        assert num_colors == 2

    def test_hexaly_k4_needs_4_colors(self, graph_k4):
        """K4 requires 4 colors with Hexaly."""
        columns = [frozenset([i]) for i in range(4)]
        num_colors, selected = solve_final_ilp(columns, 4, solver="hexaly")
        assert num_colors == 4

    def test_hexaly_matches_highs_small(self, graph_p4):
        """Hexaly and HiGHS should find same optimal on small instances."""
        columns = [
            frozenset([0]),
            frozenset([1]),
            frozenset([2]),
            frozenset([3]),
            frozenset([0, 2]),
            frozenset([1, 3]),
        ]
        result_highs = solve_final_ilp(columns, 4, solver="highs")
        result_hexaly = solve_final_ilp(columns, 4, solver="hexaly")

        # Both should find optimal 2-coloring
        assert result_highs[0] == result_hexaly[0] == 2

    def test_hexaly_with_time_limit(self, graph_p4):
        """Hexaly should work with time limit parameter."""
        columns = [
            frozenset([0]),
            frozenset([1]),
            frozenset([2]),
            frozenset([3]),
            frozenset([0, 2]),
            frozenset([1, 3]),
        ]
        num_colors, selected = solve_final_ilp(columns, 4, solver="hexaly", time_limit=60)
        assert num_colors == 2

    def test_hexaly_empty_columns(self):
        """Empty columns list should return failure."""
        num_colors, selected = solve_final_ilp([], 4, solver="hexaly")
        assert num_colors is None
        assert selected == []

    def test_hexaly_verifies_coverage(self, graph_p4):
        """Selected columns should cover all vertices exactly once."""
        columns = [
            frozenset([0]),
            frozenset([1]),
            frozenset([2]),
            frozenset([3]),
            frozenset([0, 2]),
            frozenset([1, 3]),
        ]
        num_colors, selected = solve_final_ilp(columns, 4, solver="hexaly")

        # Verify each vertex is covered exactly once
        covered = set()
        for idx in selected:
            for v in columns[idx]:
                assert v not in covered, f"Vertex {v} covered twice"
                covered.add(v)
        assert covered == {0, 1, 2, 3}
