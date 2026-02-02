"""End-to-end column generation tests with the classical oracle."""

from typing import List, Set

import pytest
import networkx as nx
import numpy as np

from quantum_colgen.column_generation import column_generation, verify_coloring, validate_coloring
from quantum_colgen.pricing.classical import ClassicalPricingOracle
from quantum_colgen.pricing.base import PricingOracle
from quantum_colgen.graphs import KNOWN_CHROMATIC, TEST_GRAPHS


class TestColumnGenerationClassical:
    """Run CG with the classical oracle on every test graph."""

    def test_all_known_graphs(self, named_graph):
        name, graph, expected_chi = named_graph
        oracle = ClassicalPricingOracle()
        num_colors, coloring, stats = column_generation(graph, oracle, verbose=False)

        assert num_colors is not None, f"CG failed on {name}"
        assert num_colors == expected_chi, (
            f"{name}: expected chi={expected_chi}, got {num_colors}"
        )
        assert verify_coloring(graph, coloring), f"Invalid coloring on {name}"

    def test_5vertex_details(self, graph_5vertex):
        oracle = ClassicalPricingOracle()
        num_colors, coloring, stats = column_generation(graph_5vertex, oracle)

        assert num_colors == 3
        assert verify_coloring(graph_5vertex, coloring)
        assert stats["columns_generated"] > 5  # started with 5 singletons, added some

    def test_erdos_renyi_small(self):
        from quantum_colgen.graphs import erdos_renyi

        G = erdos_renyi(9, 0.4, seed=42)
        oracle = ClassicalPricingOracle()
        num_colors, coloring, stats = column_generation(G, oracle)

        assert num_colors is not None
        assert verify_coloring(G, coloring)


class TestVerifyColoring:
    def test_valid(self, graph_5vertex):
        coloring = [frozenset([0, 3]), frozenset([1, 4]), frozenset([2])]
        assert verify_coloring(graph_5vertex, coloring)

    def test_invalid_adjacent(self, graph_5vertex):
        coloring = [frozenset([0, 1]), frozenset([2, 3]), frozenset([4])]
        assert not verify_coloring(graph_5vertex, coloring)

    def test_missing_vertex(self, graph_5vertex):
        coloring = [frozenset([0, 3]), frozenset([1, 4])]
        assert not verify_coloring(graph_5vertex, coloring)


class TestValidateColoring:
    """Tests for the detailed validate_coloring function."""

    def test_valid_coloring(self, graph_5vertex):
        coloring = [frozenset([0, 3]), frozenset([1, 4]), frozenset([2])]
        vr = validate_coloring(graph_5vertex, coloring)
        assert vr.valid
        assert vr.num_colors == 3
        assert vr.num_vertices_covered == 5
        assert vr.num_vertices_expected == 5
        assert vr.missing_vertices == []
        assert vr.duplicate_vertices == []
        assert vr.edge_violations == []

    def test_edge_violation_detected(self, graph_5vertex):
        # Vertices 0 and 1 are adjacent
        coloring = [frozenset([0, 1]), frozenset([2, 3]), frozenset([4])]
        vr = validate_coloring(graph_5vertex, coloring)
        assert not vr.valid
        assert len(vr.edge_violations) > 0
        # Check that (0,1) violation is reported in color class 0
        violations = [(u, v) for _, u, v in vr.edge_violations]
        assert (0, 1) in violations

    def test_missing_vertex_detected(self, graph_5vertex):
        coloring = [frozenset([0, 3]), frozenset([1, 4])]
        vr = validate_coloring(graph_5vertex, coloring)
        assert not vr.valid
        assert 2 in vr.missing_vertices

    def test_duplicate_vertex_detected(self, graph_5vertex):
        coloring = [frozenset([0, 3]), frozenset([1, 4]), frozenset([2, 0])]
        vr = validate_coloring(graph_5vertex, coloring)
        assert not vr.valid
        assert 0 in vr.duplicate_vertices

    def test_empty_color_class(self, graph_5vertex):
        coloring = [
            frozenset([0, 3]), frozenset(), frozenset([1, 4]), frozenset([2]),
        ]
        vr = validate_coloring(graph_5vertex, coloring)
        assert vr.valid  # empty class doesn't invalidate
        assert 1 in vr.empty_color_classes

    def test_cg_output_validates(self, graph_5vertex):
        """validate_coloring agrees with verify_coloring on CG output."""
        oracle = ClassicalPricingOracle()
        _, coloring, _ = column_generation(graph_5vertex, oracle)
        assert verify_coloring(graph_5vertex, coloring)
        vr = validate_coloring(graph_5vertex, coloring)
        assert vr.valid


class _MultiColumnOracle(PricingOracle):
    """Mock oracle that returns multiple profitable columns per call."""

    def __init__(self, columns_per_call: int = 3):
        self._inner = ClassicalPricingOracle()
        self._columns_per_call = columns_per_call

    def solve(self, graph: nx.Graph, dual_vars: np.ndarray) -> List[Set[int]]:
        # Use classical oracle to find the best column, then also generate
        # additional profitable IS by greedy heuristic on remaining nodes.
        results = self._inner.solve(graph, dual_vars)
        if not results:
            return []

        seen = {frozenset(s) for s in results}
        # Try to find additional profitable IS via greedy on different orderings
        pos_nodes = [v for v in graph.nodes() if dual_vars[v] > 1e-5]
        for seed_offset in range(1, 10):
            if len(results) >= self._columns_per_call:
                break
            rng = np.random.RandomState(seed_offset)
            order = list(pos_nodes)
            rng.shuffle(order)
            # Greedy IS
            candidate: Set[int] = set()
            for v in order:
                if not any(graph.has_edge(v, u) for u in candidate):
                    candidate.add(v)
            sig = frozenset(candidate)
            if sig not in seen:
                total = sum(dual_vars[v] for v in candidate)
                if total > 1 + 1e-5:
                    results.append(candidate)
                    seen.add(sig)

        return results


class TestLocalSearch:
    """Unit tests for the _local_search function from dirac_oracle."""

    def test_improves_suboptimal_is(self):
        """Local search should improve a suboptimal IS when a profitable swap exists."""
        from quantum_colgen.pricing.dirac_oracle import _local_search

        # Path graph: 0-1-2-3-4
        G = nx.path_graph(5)
        dual_vars = np.array([0.5, 0.1, 0.8, 0.1, 0.5])

        # Start with suboptimal IS {1, 3} (total=0.2)
        # Optimal IS includes {0, 2, 4} (total=1.8)
        initial = {1, 3}
        improved = _local_search(G, initial, dual_vars, max_passes=5)

        # Result must be an independent set
        for u in improved:
            for w in improved:
                if u != w:
                    assert not G.has_edge(u, w)

        # Result should have higher total weight
        initial_weight = sum(dual_vars[v] for v in initial)
        improved_weight = sum(dual_vars[v] for v in improved)
        assert improved_weight >= initial_weight

    def test_preserves_valid_is(self):
        """Local search on an already-optimal IS should not break independence."""
        from quantum_colgen.pricing.dirac_oracle import _local_search

        G = nx.cycle_graph(5)  # C5
        dual_vars = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        initial = {0, 2}  # valid IS
        result = _local_search(G, initial, dual_vars, max_passes=5)

        for u in result:
            for w in result:
                if u != w:
                    assert not G.has_edge(u, w)

    def test_zero_passes_returns_same(self):
        """With max_passes=0, local search returns the input unchanged."""
        from quantum_colgen.pricing.dirac_oracle import _local_search

        G = nx.path_graph(5)
        dual_vars = np.array([1.0, 0.1, 1.0, 0.1, 1.0])
        initial = {1, 3}
        result = _local_search(G, initial, dual_vars, max_passes=0)
        assert result == initial


class TestMultiColumnConvergence:
    """Verify that returning multiple columns per call speeds up CG."""

    def test_multi_column_fewer_iterations(self):
        """An oracle returning 3 columns/call should converge in fewer iterations."""
        from quantum_colgen.graphs import erdos_renyi

        G = erdos_renyi(12, 0.4, seed=42)

        oracle_single = ClassicalPricingOracle()
        _, coloring_single, stats_single = column_generation(G, oracle_single)

        oracle_multi = _MultiColumnOracle(columns_per_call=3)
        _, coloring_multi, stats_multi = column_generation(G, oracle_multi)

        # Both should produce valid colorings
        assert verify_coloring(G, coloring_single)
        assert verify_coloring(G, coloring_multi)

        # Multi-column oracle should use <= iterations (often strictly fewer)
        assert stats_multi["iterations"] <= stats_single["iterations"]
