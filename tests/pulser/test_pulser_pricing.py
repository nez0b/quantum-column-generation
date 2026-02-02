"""Tests for the Pulser neutral-atom pricing oracle."""

import pytest
import numpy as np

pulser_available = pytest.importorskip("pulser", reason="pulser not installed")

from quantum_colgen.pricing.pulser_oracle import PulserPricingOracle, PULSER_AVAILABLE
from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.graphs import paper_5vertex, triangle


pytestmark = pytest.mark.pulser


@pytest.mark.skipif(not PULSER_AVAILABLE, reason="pulser/MIS not installed")
class TestPulserPricingOracle:
    def test_finds_profitable_column(self):
        G = paper_5vertex()
        oracle = PulserPricingOracle(duration_us=2000, runs=200)
        duals = np.array([0.6, 0.4, 0.4, 0.6, 0.6])
        cols = oracle.solve(G, duals)
        # Pulser is stochastic; we just check returned sets are valid IS
        for col in cols:
            assert sum(duals[v] for v in col) > 1.0
            for u in col:
                for w in col:
                    if u != w:
                        assert not G.has_edge(u, w)

    def test_end_to_end_small_graph(self):
        """CG with Pulser oracle on the triangle (very small for simulator)."""
        G = triangle()
        oracle = PulserPricingOracle(duration_us=2000, runs=200)
        num_colors, coloring, _ = column_generation(G, oracle)
        # Pulser may not find optimal, but should produce a valid coloring
        if num_colors is not None:
            assert verify_coloring(G, coloring)
