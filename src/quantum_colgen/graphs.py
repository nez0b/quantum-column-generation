"""Test graph generators for column generation benchmarks."""

import networkx as nx


def paper_5vertex() -> nx.Graph:
    """The 5-vertex graph from arXiv:2301.02637. Chromatic number = 3."""
    G = nx.Graph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])
    return G


def triangle() -> nx.Graph:
    """K3 triangle. Chromatic number = 3."""
    return nx.cycle_graph(3)


def complete_k4() -> nx.Graph:
    """K4 complete graph. Chromatic number = 4."""
    return nx.complete_graph(4)


def path_p4() -> nx.Graph:
    """P4 path graph. Chromatic number = 2."""
    return nx.path_graph(4)


def cycle_c5() -> nx.Graph:
    """C5 odd cycle. Chromatic number = 3."""
    return nx.cycle_graph(5)


def wheel_w5() -> nx.Graph:
    """Wheel graph W5 (6 nodes including center). Chromatic number = 4."""
    return nx.wheel_graph(6)


def erdos_renyi(n: int, p: float, seed: int = 42) -> nx.Graph:
    """Erdos-Renyi random graph G(n,p)."""
    return nx.gnp_random_graph(n, p, seed=seed)


KNOWN_CHROMATIC = {
    "paper_5vertex": 3,
    "triangle": 3,
    "complete_k4": 4,
    "path_p4": 2,
    "cycle_c5": 3,
    "wheel_w5": 4,
}

TEST_GRAPHS = {
    "paper_5vertex": paper_5vertex,
    "triangle": triangle,
    "complete_k4": complete_k4,
    "path_p4": path_p4,
    "cycle_c5": cycle_c5,
    "wheel_w5": wheel_w5,
}
