"""Manim Graph construction from a PSPData record.

Reuses the cached Kamada-Kawai layout from the beamer deck so viewers
who saw the static slides recognize the graph instantly.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from manim import Animation, Graph, ORIGIN

from .psp_loader import PSPData
from .qci_colors import QCI_TEAL
from .theme import EDGE_COLOR, FG_PRIMARY, INACTIVE_VERTEX


def build_psp_graph(
    psp: PSPData,
    scale: float = 2.8,
    vertex_radius: float = 0.13,
    position=ORIGIN,
) -> Graph:
    """Graph mobject with vertices placed via the cached PSP layout."""
    layout3d = {
        v: np.array([x * scale, y * scale, 0.0])
        for v, (x, y) in psp.layout.items()
    }
    g = Graph(
        vertices=list(psp.node_list),
        edges=[tuple(e) for e in psp.graph_edges],
        layout=layout3d,
        vertex_config={
            "radius": vertex_radius,
            "fill_color": FG_PRIMARY,
            "fill_opacity": 0.95,
            "stroke_color": FG_PRIMARY,
            "stroke_width": 1.0,
        },
        edge_config={
            "stroke_color": EDGE_COLOR,
            "stroke_width": 1.2,
        },
        labels=False,
    )
    g.move_to(position)
    return g


def dim_inactive_anims(
    graph: Graph,
    active_nodes: Iterable[int],
    inactive_opacity: float = 0.15,
    active_color=QCI_TEAL,
) -> List[Animation]:
    """Fade inactive vertices and give active ones a distinct teal fill."""
    active = set(active_nodes)
    anims: List[Animation] = []
    for v, vmob in graph.vertices.items():
        if v in active:
            anims.append(vmob.animate.set_fill(active_color, opacity=1.0))
        else:
            anims.append(vmob.animate.set_fill(INACTIVE_VERTEX, opacity=max(inactive_opacity, 0.55)))
    return anims


def pulse_active_anims(
    graph: Graph,
    active_nodes: Iterable[int],
    scale: float = 1.4,
) -> List[Animation]:
    """Scale up active vertices (for a brief attention-getting pulse)."""
    return [graph.vertices[v].animate.scale(scale) for v in active_nodes]


def unpulse_anims(
    graph: Graph,
    active_nodes: Iterable[int],
    scale: float = 1.4,
) -> List[Animation]:
    return [graph.vertices[v].animate.scale(1.0 / scale) for v in active_nodes]
