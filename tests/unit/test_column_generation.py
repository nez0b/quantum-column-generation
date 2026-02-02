"""End-to-end column generation tests with the classical oracle."""

import pytest

from quantum_colgen.column_generation import column_generation, verify_coloring, validate_coloring
from quantum_colgen.pricing.classical import ClassicalPricingOracle
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
