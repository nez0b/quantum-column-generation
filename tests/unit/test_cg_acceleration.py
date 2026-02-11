"""Tests for Bomze regularization and CG acceleration features."""

import networkx as nx
import numpy as np
import pytest

from quantum_colgen.pricing.dirac_oracle import _construct_gibbons_matrix
from quantum_colgen.column_generation import column_generation, verify_coloring
from quantum_colgen.pricing.classical import ClassicalPricingOracle
from quantum_colgen.pricing.classical_lp import ClassicalLPPricingOracle
from quantum_colgen.graphs import erdos_renyi


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_complement_and_weights():
    """Return a small complement graph and weights for Gibbons matrix tests."""
    # 4-node path: 0-1-2-3, complement edges: (0,2), (0,3), (1,3)
    G = nx.path_graph(4)
    complement = nx.complement(G)
    weights = {0: 1.0, 1: 2.0, 2: 1.5, 3: 0.5}
    return complement, weights


# ---------------------------------------------------------------------------
# TestGibbonsRegularization
# ---------------------------------------------------------------------------

class TestGibbonsRegularization:

    def test_c_zero_unchanged(self):
        """B with c=0 should be identical to B without regularization."""
        complement, weights = _small_complement_and_weights()
        B_base = _construct_gibbons_matrix(complement, weights)
        B_zero = _construct_gibbons_matrix(complement, weights, regularization_c=0.0)
        np.testing.assert_array_equal(B_base, B_zero)

    def test_diagonal_decremented(self):
        """Diagonal entries should decrease by c when regularization applied."""
        complement, weights = _small_complement_and_weights()
        c = 0.1
        B_base = _construct_gibbons_matrix(complement, weights)
        B_reg = _construct_gibbons_matrix(complement, weights, regularization_c=c)

        n = B_base.shape[0]
        for i in range(n):
            assert B_reg[i, i] == pytest.approx(B_base[i, i] - c, abs=1e-12)

    def test_off_diagonal_unchanged(self):
        """Off-diagonal entries should be unaffected by regularization."""
        complement, weights = _small_complement_and_weights()
        c = 0.2
        B_base = _construct_gibbons_matrix(complement, weights)
        B_reg = _construct_gibbons_matrix(complement, weights, regularization_c=c)

        n = B_base.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert B_reg[i, j] == pytest.approx(B_base[i, j], abs=1e-12)

    def test_positive_diagonal_with_valid_c(self):
        """When c < min(1/w), all diagonal entries remain positive."""
        complement, weights = _small_complement_and_weights()
        min_inv_w = min(1.0 / w for w in weights.values())  # 1/2.0 = 0.5
        c = 0.3  # < 0.5
        B_reg = _construct_gibbons_matrix(complement, weights, regularization_c=c)

        n = B_reg.shape[0]
        for i in range(n):
            assert B_reg[i, i] > 0, f"Diagonal B[{i},{i}] = {B_reg[i, i]} <= 0"

    def test_large_c_clamped(self):
        """When c >= min(1/w), c should be clamped to keep diagonal positive."""
        complement, weights = _small_complement_and_weights()
        # min(1/w) = 1/2.0 = 0.5; asking for c=10.0 should be clamped
        B_reg = _construct_gibbons_matrix(complement, weights, regularization_c=10.0)

        n = B_reg.shape[0]
        for i in range(n):
            assert B_reg[i, i] > 0, f"Diagonal B[{i},{i}] = {B_reg[i, i]} <= 0 after clamping"

    def test_clamped_c_is_0_9_times_min_diag(self):
        """Verify clamped c = 0.9 * min(B[i,i])."""
        complement, weights = _small_complement_and_weights()
        B_base = _construct_gibbons_matrix(complement, weights)
        min_diag = min(B_base[i, i] for i in range(B_base.shape[0]))
        expected_c = 0.9 * min_diag

        B_reg = _construct_gibbons_matrix(complement, weights, regularization_c=10.0)
        n = B_base.shape[0]
        for i in range(n):
            assert B_reg[i, i] == pytest.approx(B_base[i, i] - expected_c, abs=1e-12)


# ---------------------------------------------------------------------------
# TestDualSmoothing
# ---------------------------------------------------------------------------

class TestDualSmoothing:

    def test_alpha_zero_passthrough(self):
        """With alpha=0 (or None), CG should produce same result as baseline."""
        G = erdos_renyi(10, 0.4, seed=42)
        oracle = ClassicalPricingOracle()

        chi_base, col_base, stats_base = column_generation(G, oracle)
        chi_smooth, col_smooth, stats_smooth = column_generation(
            G, oracle, dual_smoothing_alpha=0.0,
        )

        assert chi_base == chi_smooth
        assert verify_coloring(G, col_base)
        assert verify_coloring(G, col_smooth)

    def test_smoothing_convergence(self):
        """CG with moderate smoothing should still converge to a valid coloring."""
        G = erdos_renyi(12, 0.4, seed=42)
        oracle = ClassicalPricingOracle()

        chi, coloring, stats = column_generation(
            G, oracle, dual_smoothing_alpha=0.3,
        )

        assert chi is not None
        assert verify_coloring(G, coloring)

    def test_stats_include_rmp_obj_trace(self):
        """Stats should include rmp_obj_trace with one entry per iteration."""
        G = erdos_renyi(8, 0.4, seed=42)
        oracle = ClassicalPricingOracle()

        _, _, stats = column_generation(G, oracle)

        assert "rmp_obj_trace" in stats
        assert len(stats["rmp_obj_trace"]) > 0
        # LP bound should be monotonically non-increasing (minimization: adding
        # columns can only decrease or maintain the LP relaxation value)
        trace = stats["rmp_obj_trace"]
        for i in range(1, len(trace)):
            assert trace[i] <= trace[i - 1] + 1e-6


# ---------------------------------------------------------------------------
# TestSubproblemAging
# ---------------------------------------------------------------------------

class TestSubproblemAging:

    def test_convergence_with_aging(self):
        """CG with conservative aging threshold should still converge."""
        G = erdos_renyi(12, 0.4, seed=42)
        oracle = ClassicalPricingOracle()

        chi, coloring, stats = column_generation(
            G, oracle, subproblem_aging_threshold=0.01,
        )

        assert chi is not None
        assert verify_coloring(G, coloring)

    def test_skipped_calls_tracked(self):
        """Stats should track oracle_calls_skipped."""
        G = erdos_renyi(10, 0.4, seed=42)
        oracle = ClassicalPricingOracle()

        _, _, stats = column_generation(
            G, oracle, subproblem_aging_threshold=0.01,
        )

        assert "oracle_calls_skipped" in stats
        # With the classical oracle, duals may not change much near convergence
        assert isinstance(stats["oracle_calls_skipped"], int)

    def test_large_threshold_skips_more(self):
        """A larger aging threshold should skip more oracle calls."""
        G = erdos_renyi(12, 0.4, seed=42)
        oracle = ClassicalPricingOracle()

        _, _, stats_conservative = column_generation(
            G, oracle, subproblem_aging_threshold=0.001,
        )
        _, _, stats_aggressive = column_generation(
            G, oracle, subproblem_aging_threshold=0.5,
        )

        assert stats_aggressive["oracle_calls_skipped"] >= stats_conservative["oracle_calls_skipped"]


# ---------------------------------------------------------------------------
# TestCombined
# ---------------------------------------------------------------------------

class TestCombinedAcceleration:

    def test_smoothing_plus_aging(self):
        """CG with both smoothing and aging should still produce valid coloring."""
        G = erdos_renyi(12, 0.4, seed=42)
        oracle = ClassicalPricingOracle()

        chi, coloring, stats = column_generation(
            G, oracle,
            dual_smoothing_alpha=0.3,
            subproblem_aging_threshold=0.05,
        )

        assert chi is not None
        assert verify_coloring(G, coloring)

    def test_lp_oracle_with_smoothing(self):
        """LP oracle + smoothing should work end-to-end."""
        G = erdos_renyi(15, 0.3, seed=42)
        oracle = ClassicalLPPricingOracle()

        chi, coloring, stats = column_generation(
            G, oracle, dual_smoothing_alpha=0.3,
        )

        assert chi is not None
        assert verify_coloring(G, coloring)
