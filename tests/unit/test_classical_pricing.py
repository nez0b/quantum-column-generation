"""Tests for the classical MILP pricing oracle."""

import numpy as np
import pytest

from quantum_colgen.pricing.classical import ClassicalPricingOracle


class TestClassicalPricingOracle:
    def setup_method(self):
        self.oracle = ClassicalPricingOracle()

    def test_finds_profitable_column_5vertex(self, graph_5vertex):
        """With uniform duals > 1/|IS|, oracle should find a profitable IS."""
        # Duals that make {0,3} or {0,4} profitable (sum > 1)
        duals = np.array([0.6, 0.4, 0.4, 0.6, 0.6])
        cols = self.oracle.solve(graph_5vertex, duals)
        assert len(cols) >= 1
        for col in cols:
            assert sum(duals[v] for v in col) > 1.0
            # Each returned set must be independent
            for u in col:
                for w in col:
                    if u != w:
                        assert not graph_5vertex.has_edge(u, w)

    def test_no_profitable_column(self, graph_triangle):
        """On K3 with duals=1, each vertex alone has weight 1 (not > 1)."""
        duals = np.array([1.0, 1.0, 1.0])
        cols = self.oracle.solve(graph_triangle, duals)
        assert cols == []

    def test_zero_duals(self, graph_5vertex):
        duals = np.zeros(5)
        cols = self.oracle.solve(graph_5vertex, duals)
        assert cols == []

    def test_edgeless_graph(self):
        """Edgeless graph: all nodes form one IS."""
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(range(3))
        duals = np.array([0.5, 0.5, 0.5])
        cols = self.oracle.solve(G, duals)
        assert len(cols) == 1
        assert cols[0] == {0, 1, 2}
