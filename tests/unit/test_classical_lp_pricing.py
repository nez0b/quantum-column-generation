"""Tests for the Classical LP relaxation pricing oracle."""

import pytest
import numpy as np

from quantum_colgen.pricing.classical_lp import ClassicalLPPricingOracle
from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.graphs import paper_5vertex, triangle, path_p4, complete_k4


class TestClassicalLPPricingOracle:
    """Tests for ClassicalLPPricingOracle."""

    def test_finds_profitable_column_5vertex(self):
        """LP oracle finds profitable column on 5-vertex graph."""
        G = paper_5vertex()
        oracle = ClassicalLPPricingOracle()
        # Edges: (0,1), (0,2), (1,2), (1,3), (2,4), (3,4)
        # {0, 3} is an IS with profit 0.6 + 0.6 = 1.2 > 1
        duals = np.array([0.6, 0.4, 0.4, 0.6, 0.4])
        cols = oracle.solve(G, duals)
        # Should find at least one profitable IS
        assert len(cols) >= 1
        for col in cols:
            total = sum(duals[v] for v in col)
            assert total > 1.0
            # Verify independence
            for u in col:
                for w in col:
                    if u != w:
                        assert not G.has_edge(u, w)

    def test_no_profitable_column(self):
        """LP oracle returns empty when no profitable column exists."""
        G = triangle()
        oracle = ClassicalLPPricingOracle()
        # Duals sum to 1 for any single node, no pair is independent
        duals = np.array([0.5, 0.5, 0.5])
        cols = oracle.solve(G, duals)
        # No column with profit > 1 exists
        assert cols == []

    def test_zero_duals(self):
        """LP oracle returns empty when all duals are zero."""
        G = paper_5vertex()
        oracle = ClassicalLPPricingOracle()
        duals = np.zeros(5)
        cols = oracle.solve(G, duals)
        assert cols == []

    def test_edgeless_graph(self):
        """LP oracle handles edgeless graph (all nodes form one IS)."""
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        oracle = ClassicalLPPricingOracle()
        duals = np.array([0.5, 0.5, 0.5])  # sum = 1.5 > 1
        cols = oracle.solve(G, duals)
        assert len(cols) >= 1
        # The full set {0, 1, 2} should be returned
        assert any(col == {0, 1, 2} for col in cols)

    def test_multi_prune_extracts_more_columns(self):
        """Multi-prune option extracts more columns than baseline."""
        G = paper_5vertex()
        duals = np.array([0.6, 0.4, 0.4, 0.6, 0.6])

        oracle_single = ClassicalLPPricingOracle(
            multi_prune=False,
            randomized_rounding=False,
        )
        oracle_multi = ClassicalLPPricingOracle(
            multi_prune=True,
            num_random_prune_trials=3,
            randomized_rounding=False,
        )

        cols_single = oracle_single.solve(G, duals)
        cols_multi = oracle_multi.solve(G, duals)

        # All columns should be valid IS
        for col in cols_multi:
            for u in col:
                for w in col:
                    if u != w:
                        assert not G.has_edge(u, w)

        # Multi-prune should find at least as many
        assert len(cols_multi) >= len(cols_single)

    def test_randomized_rounding_produces_valid_is(self):
        """Randomized rounding produces valid independent sets."""
        G = paper_5vertex()
        oracle = ClassicalLPPricingOracle(
            randomized_rounding=True,
            num_random_rounds=10,
            random_seed=42,
        )
        duals = np.array([0.6, 0.4, 0.4, 0.6, 0.6])
        cols = oracle.solve(G, duals)

        for col in cols:
            total = sum(duals[v] for v in col)
            assert total > 1.0
            for u in col:
                for w in col:
                    if u != w:
                        assert not G.has_edge(u, w)

    def test_end_to_end_path(self):
        """CG with LP oracle on P4 (chi=2)."""
        G = path_p4()
        oracle = ClassicalLPPricingOracle()
        num_colors, coloring, stats = column_generation(G, oracle)
        assert num_colors is not None
        assert verify_coloring(G, coloring)
        assert num_colors == 2

    def test_end_to_end_5vertex(self):
        """CG with LP oracle on paper's 5-vertex graph (chi=3)."""
        G = paper_5vertex()
        oracle = ClassicalLPPricingOracle()
        num_colors, coloring, stats = column_generation(G, oracle)
        assert num_colors is not None
        assert verify_coloring(G, coloring)
        assert num_colors == 3

    def test_end_to_end_complete_k4(self):
        """CG with LP oracle on K4 (chi=4)."""
        G = complete_k4()
        oracle = ClassicalLPPricingOracle()
        num_colors, coloring, stats = column_generation(G, oracle)
        assert num_colors is not None
        assert verify_coloring(G, coloring)
        assert num_colors == 4

    def test_timer_records_calls(self):
        """Oracle timer records API calls and extraction time."""
        G = paper_5vertex()
        oracle = ClassicalLPPricingOracle()
        duals = np.array([0.6, 0.4, 0.4, 0.6, 0.6])
        oracle.solve(G, duals)

        assert oracle.timer.num_calls == 1
        assert oracle.timer.total_api_seconds > 0

    def test_custom_thresholds(self):
        """Custom thresholds are respected."""
        G = paper_5vertex()
        oracle = ClassicalLPPricingOracle(
            support_thresholds=[0.5, 0.9],
            multi_prune=False,
            randomized_rounding=False,
        )
        assert oracle.support_thresholds == [0.5, 0.9]
        duals = np.array([0.6, 0.4, 0.4, 0.6, 0.6])
        cols = oracle.solve(G, duals)
        # Should still produce valid results
        for col in cols:
            for u in col:
                for w in col:
                    if u != w:
                        assert not G.has_edge(u, w)
