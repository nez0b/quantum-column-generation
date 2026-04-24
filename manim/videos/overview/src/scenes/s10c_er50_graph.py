"""Slide 10c: ER(50,0.3) Graph Visualization."""

import json
from pathlib import Path

from manim import *
from manim_slides import Slide


class ER50GraphSlide(Slide):
    """Shows the ER(50,0.3) graph before coloring."""

    def construct(self):
        # Load graph data
        data_path = Path(__file__).parent.parent / "data" / "graph_colorings.json"
        if data_path.exists():
            with open(data_path) as f:
                data = json.load(f)
            graph_data = data.get("ER(50,0.3)", {})
            n_nodes = graph_data.get("n_nodes", 50)
            n_edges = graph_data.get("n_edges", 368)
            edges = graph_data.get("edges", [])
        else:
            # Fallback: generate graph on the fly
            import networkx as nx
            G = nx.gnp_random_graph(50, 0.3, seed=42)
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            edges = list(G.edges())

        # Title
        title = Text(
            f"ER(50, 0.3) - {n_nodes} nodes, {n_edges} edges",
            font_size=36,
            weight=BOLD,
        ).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.next_slide()

        # Create manim graph
        vertices = list(range(n_nodes))
        edge_tuples = [tuple(e) for e in edges]

        g = Graph(
            vertices,
            edge_tuples,
            layout="kamada_kawai",
            layout_scale=3.0,
            labels=False,
            vertex_config={
                "radius": 0.07,
                "fill_color": WHITE,
                "fill_opacity": 0.9,
                "stroke_width": 1,
                "stroke_color": GRAY,
            },
            edge_config={
                "stroke_width": 0.4,
                "stroke_color": GRAY_B,
            },
        ).shift(DOWN * 0.3)

        self.play(Create(g), run_time=2.0)
        self.next_slide()

        # Subtitle with challenge
        subtitle = Text(
            "A larger, denser challenge for graph coloring",
            font_size=26,
            color=GRAY_A,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(subtitle))
        self.next_slide()
