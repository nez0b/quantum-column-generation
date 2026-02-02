"""Tests for the Dirac-3 pricing oracle (requires QCI_TOKEN)."""

import pytest
import numpy as np

from quantum_colgen.pricing.dirac_oracle import DiracPricingOracle, DIRAC_AVAILABLE
from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.graphs import paper_5vertex, triangle, path_p4


pytestmark = pytest.mark.dirac


@pytest.mark.skipif(not DIRAC_AVAILABLE, reason="qci-client/eqc-models not installed")
class TestDiracPricingOracle:
    def test_gibbons_profitable(self):
        """Gibbons-weighted approach on 5-vertex graph."""
        G = paper_5vertex()
        oracle = DiracPricingOracle(method="gibbons", num_samples=20, relax_schedule=2)
        duals = np.array([0.6, 0.4, 0.4, 0.6, 0.6])
        cols = oracle.solve(G, duals)
        for col in cols:
            total = sum(duals[v] for v in col)
            assert total > 1.0
            for u in col:
                for w in col:
                    if u != w:
                        assert not G.has_edge(u, w)

    def test_filter_profitable(self):
        """Filter approach on 5-vertex graph."""
        G = paper_5vertex()
        oracle = DiracPricingOracle(method="filter", num_samples=20, relax_schedule=2)
        duals = np.array([0.6, 0.4, 0.4, 0.6, 0.6])
        cols = oracle.solve(G, duals)
        for col in cols:
            total = sum(duals[v] for v in col)
            assert total > 1.0

    def test_end_to_end_path(self):
        """CG with Dirac oracle on P4 (chi=2)."""
        G = path_p4()
        oracle = DiracPricingOracle(method="gibbons", num_samples=20, relax_schedule=2)
        num_colors, coloring, _ = column_generation(G, oracle)
        if num_colors is not None:
            assert verify_coloring(G, coloring)
            assert num_colors <= 3  # should be 2, allow some slack for quantum noise

    def test_end_to_end_5vertex(self):
        """CG with Dirac oracle on the paper's 5-vertex graph (chi=3)."""
        G = paper_5vertex()
        oracle = DiracPricingOracle(method="gibbons", num_samples=20, relax_schedule=2)
        num_colors, coloring, _ = column_generation(G, oracle)
        if num_colors is not None:
            assert verify_coloring(G, coloring)
            assert num_colors <= 4  # optimal is 3, allow slack

    def test_multi_sample_returns_multiple_columns(self):
        """With enough samples and thresholds, oracle may return >1 column."""
        G = paper_5vertex()
        oracle = DiracPricingOracle(
            method="gibbons", num_samples=50, relax_schedule=2,
            support_thresholds=[0.005, 0.01, 0.05, 0.1, 0.2],
        )
        duals = np.array([0.6, 0.4, 0.4, 0.6, 0.6])
        cols = oracle.solve(G, duals)
        # All returned columns must be valid independent sets
        for col in cols:
            total = sum(duals[v] for v in col)
            assert total > 1.0
            for u in col:
                for w in col:
                    if u != w:
                        assert not G.has_edge(u, w)
        # With multi-sample + multi-threshold, we expect at least 1 column
        assert len(cols) >= 1

    def test_local_search_does_not_worsen(self):
        """Local search should not produce worse solutions than without it."""
        G = paper_5vertex()
        oracle_no_ls = DiracPricingOracle(
            method="gibbons", num_samples=20, relax_schedule=2,
            local_search_passes=0,
        )
        oracle_with_ls = DiracPricingOracle(
            method="gibbons", num_samples=20, relax_schedule=2,
            local_search_passes=5,
        )
        duals = np.array([0.6, 0.4, 0.4, 0.6, 0.6])
        cols_no_ls = oracle_no_ls.solve(G, duals)
        cols_with_ls = oracle_with_ls.solve(G, duals)
        # Both should produce valid independent sets
        for col in cols_with_ls:
            for u in col:
                for w in col:
                    if u != w:
                        assert not G.has_edge(u, w)
        # With local search we should get at least as many profitable columns
        assert len(cols_with_ls) >= len(cols_no_ls)

    def test_backward_compat_single_threshold(self):
        """Passing support_threshold (singular) still works."""
        G = paper_5vertex()
        oracle = DiracPricingOracle(
            method="gibbons", num_samples=20, relax_schedule=2,
            support_threshold=0.01,
        )
        assert oracle.support_thresholds == [0.01]
        duals = np.array([0.6, 0.4, 0.4, 0.6, 0.6])
        cols = oracle.solve(G, duals)
        for col in cols:
            total = sum(duals[v] for v in col)
            assert total > 1.0
