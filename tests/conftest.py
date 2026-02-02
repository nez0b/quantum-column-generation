"""Shared test fixtures for quantum-colgen."""

import pytest
import networkx as nx

from quantum_colgen.graphs import (
    paper_5vertex,
    triangle,
    complete_k4,
    path_p4,
    cycle_c5,
    wheel_w5,
    KNOWN_CHROMATIC,
)


@pytest.fixture
def graph_5vertex():
    """5-vertex graph from the paper (chromatic number 3)."""
    return paper_5vertex()


@pytest.fixture
def graph_triangle():
    return triangle()


@pytest.fixture
def graph_k4():
    return complete_k4()


@pytest.fixture
def graph_p4():
    return path_p4()


@pytest.fixture
def graph_c5():
    return cycle_c5()


@pytest.fixture
def graph_w5():
    return wheel_w5()


@pytest.fixture(params=list(KNOWN_CHROMATIC.keys()))
def named_graph(request):
    """Parametrized fixture yielding (name, graph, expected_chromatic_number)."""
    from quantum_colgen.graphs import TEST_GRAPHS

    name = request.param
    return name, TEST_GRAPHS[name](), KNOWN_CHROMATIC[name]
