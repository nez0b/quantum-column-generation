"""Slide 10b: ER(40,0.3) Coloring Comparison."""

import json
from pathlib import Path

from manim import *
from manim_slides import Slide

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.colors import get_graph_color


class ER40ColoringsSlide(Slide):
    """Shows side-by-side coloring comparison for ER(40,0.3)."""

    def construct(self):
        # Load graph data
        data_path = Path(__file__).parent.parent / "data" / "graph_colorings.json"
        if data_path.exists():
            with open(data_path) as f:
                data = json.load(f)
            graph_data = data.get("ER(40,0.3)", {})
            n_nodes = graph_data.get("n_nodes", 40)
            edges = graph_data.get("edges", [])
            colorings = graph_data.get("colorings", {})
        else:
            # Fallback
            import networkx as nx
            G = nx.gnp_random_graph(40, 0.3, seed=42)
            n_nodes = G.number_of_nodes()
            edges = list(G.edges())
            colorings = {}

        # Title
        title = Text("ER(40, 0.3) - Coloring Comparison", font_size=36, weight=BOLD).to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.next_slide()

        # Create three graphs side by side
        vertices = list(range(n_nodes))
        edge_tuples = [tuple(e) for e in edges]

        # Method order: Greedy, LP, Dirac
        methods = [
            ("greedy", "Greedy", YELLOW),
            ("lp", "LP", ORANGE),
            ("dirac", "Dirac", GREEN),
        ]

        x_positions = [-4.5, 0, 4.5]
        scale_factor = 0.35

        graphs = []
        labels = []

        for i, (method_key, method_name, label_color) in enumerate(methods):
            # Get coloring data
            method_data = colorings.get(method_key, {})
            chi = method_data.get("chi", "?")
            color_classes = method_data.get("color_classes", [])

            # Build vertex color map
            vertex_colors = {}
            for class_idx, nodes in enumerate(color_classes):
                color = get_graph_color(class_idx)
                for node in nodes:
                    vertex_colors[node] = color

            # Create graph with colored vertices
            vertex_config = {
                v: {
                    "radius": 0.06,
                    "fill_color": vertex_colors.get(v, WHITE),
                    "fill_opacity": 0.95,
                    "stroke_width": 0.5,
                    "stroke_color": GRAY,
                }
                for v in vertices
            }

            g = Graph(
                vertices,
                edge_tuples,
                layout="kamada_kawai",
                layout_scale=2.5,
                labels=False,
                vertex_config=vertex_config,
                edge_config={
                    "stroke_width": 0.3,
                    "stroke_color": GRAY_C,
                },
            ).scale(scale_factor).shift(RIGHT * x_positions[i] + DOWN * 0.2)

            graphs.append(g)

            # Create label
            chi_text = f"{chi}" if chi != "?" else "?"
            label = VGroup(
                Text(method_name, font_size=22, color=label_color, weight=BOLD),
                MathTex(rf"\chi = {chi_text}", font_size=28, color=label_color),
            ).arrange(DOWN, buff=0.1).next_to(g, DOWN, buff=0.3)
            labels.append(label)

        # Animate graphs appearing
        self.play(*[Create(g) for g in graphs], run_time=2.0)
        self.next_slide()

        # Show labels
        self.play(*[FadeIn(lbl) for lbl in labels])
        self.next_slide()

        # Highlight Dirac winner with box
        dirac_box = SurroundingRectangle(
            VGroup(graphs[2], labels[2]),
            color=GREEN,
            buff=0.15,
            corner_radius=0.1,
            stroke_width=3,
        )
        self.play(Create(dirac_box))
        self.next_slide()

        # Victory text
        victory_text = Text(
            "Dirac achieves 6 colors - 2 fewer than classical methods!",
            font_size=24,
            color=GREEN,
        ).to_edge(DOWN, buff=0.4)
        self.play(FadeIn(victory_text))
        self.next_slide()
