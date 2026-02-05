"""Graph utilities for creating and animating graphs."""

from manim import *
import networkx as nx
from typing import List, Set, Tuple, Dict, Optional

from .colors import get_graph_color, GRAPH_COLORS


def create_example_graph() -> Tuple[List[int], List[Tuple[int, int]]]:
    """Create a small example graph for demonstrations.

    Returns a 5-vertex graph that requires 3 colors.
    """
    vertices = [0, 1, 2, 3, 4]
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]
    return vertices, edges


def create_petersen_subgraph() -> Tuple[List[int], List[Tuple[int, int]]]:
    """Create a small subgraph inspired by the Petersen graph."""
    vertices = list(range(6))
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)]
    return vertices, edges


def get_valid_coloring(
    vertices: List[int], edges: List[Tuple[int, int]]
) -> Dict[int, int]:
    """Compute a valid graph coloring using greedy algorithm.

    Returns a dict mapping vertex -> color index.
    """
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)
    return nx.greedy_color(G, strategy="largest_first")


def highlight_independent_set(
    graph: Graph,
    independent_set: Set[int],
    highlight_color: ManimColor = YELLOW,
    scale: float = 1.3,
) -> List[Animation]:
    """Create animations to highlight an independent set in a graph.

    Returns list of animations that can be played together.
    """
    animations = []
    for v in independent_set:
        animations.append(graph.vertices[v].animate.scale(scale))
        animations.append(
            graph.vertices[v].animate.set_stroke(highlight_color, width=4)
        )
    return animations


def color_vertices(
    graph: Graph,
    coloring: Dict[int, int],
    opacity: float = 1.0,
) -> List[Animation]:
    """Create animations to color vertices according to a coloring dict.

    Returns list of animations that can be played together.
    """
    animations = []
    for vertex, color_idx in coloring.items():
        color = get_graph_color(color_idx)
        animations.append(
            graph.vertices[vertex].animate.set_fill(color, opacity=opacity)
        )
    return animations


def create_column_vector(
    independent_set: Set[int],
    n_vertices: int,
    position: np.ndarray = ORIGIN,
    scale: float = 0.6,
) -> VGroup:
    """Create a visual column vector from an independent set.

    The vector has 1s for vertices in the IS and 0s otherwise.
    """
    entries = []
    for v in range(n_vertices):
        val = "1" if v in independent_set else "0"
        entry = MathTex(val)
        entries.append(entry)

    # Arrange vertically
    column = VGroup(*entries).arrange(DOWN, buff=0.1)

    # Add brackets
    left_bracket = MathTex(r"\begin{bmatrix}").scale(len(entries) * 0.3)
    right_bracket = MathTex(r"\end{bmatrix}").scale(len(entries) * 0.3)

    # Position brackets - simplified approach using VGroup
    vector_group = VGroup(column).scale(scale).move_to(position)

    return vector_group


def is_independent_set(edges: List[Tuple[int, int]], vertex_set: Set[int]) -> bool:
    """Check if a set of vertices forms an independent set."""
    for u, v in edges:
        if u in vertex_set and v in vertex_set:
            return False
    return True


def find_all_maximal_is(
    vertices: List[int], edges: List[Tuple[int, int]]
) -> List[Set[int]]:
    """Find all maximal independent sets in a small graph.

    Only use for small graphs (n <= 10) as this is exponential.
    """
    G = nx.Graph()
    G.add_nodes_from(vertices)
    G.add_edges_from(edges)

    # Use networkx to find maximal independent sets
    from networkx.algorithms.mis import maximal_independent_set

    # Find multiple by starting from different vertices
    all_mis = []
    for start in vertices:
        try:
            mis = set(maximal_independent_set(G, nodes=[start]))
            if mis not in all_mis:
                all_mis.append(mis)
        except nx.NetworkXUnfeasible:
            pass

    return all_mis
