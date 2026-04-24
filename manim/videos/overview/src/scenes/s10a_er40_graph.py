"""Slide 10a: ER(40,0.3) Graph Visualization."""

import json
from pathlib import Path

from manim import *
from manim_slides import Slide


class ER40GraphSlide(Slide):
    """Shows the ER(40,0.3) graph before coloring."""

    def construct(self):
        # Load graph data
        data_path = Path(__file__).parent.parent / "data" / "graph_colorings.json"
        if data_path.exists():
            with open(data_path) as f:
                data = json.load(f)
            graph_data = data.get("ER(40,0.3)", {})
            n_nodes = graph_data.get("n_nodes", 40)
            n_edges = graph_data.get("n_edges", 244)
            edges = graph_data.get("edges", [])
        else:
            # Fallback: generate graph on the fly
            import networkx as nx
            G = nx.gnp_random_graph(40, 0.3, seed=42)
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            edges = list(G.edges())

        # Title
        title = Text(
            f"ER(40, 0.3) - {n_nodes} nodes, {n_edges} edges",
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
            labels=False,  # Too cluttered with 40 nodes
            vertex_config={
                "radius": 0.08,
                "fill_color": WHITE,
                "fill_opacity": 0.9,
                "stroke_width": 1,
                "stroke_color": GRAY,
            },
            edge_config={
                "stroke_width": 0.5,
                "stroke_color": GRAY_B,
            },
        ).shift(DOWN * 0.3)

        self.play(Create(g), run_time=2.0)
        self.next_slide()

        # Question text
        question = Text(
            "Can we color this with fewer than 8 colors?",
            font_size=28,
            color=YELLOW,
        ).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(question))
        self.next_slide()
