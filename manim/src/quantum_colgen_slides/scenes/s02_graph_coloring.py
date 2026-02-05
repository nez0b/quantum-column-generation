"""Slide 2: Graph Coloring Problem."""

from manim import *
from manim_slides import Slide

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from components.colors import GRAPH_RED, GRAPH_BLUE, GRAPH_GREEN


class GraphColoringSlide(Slide):
    """Explains the graph coloring problem visually."""

    def construct(self):
        # Title
        title = Text("Graph Coloring Problem", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.next_slide()

        # Create graph
        vertices = [0, 1, 2, 3, 4]
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]

        g = Graph(
            vertices,
            edges,
            layout="kamada_kawai",
            labels=True,
            vertex_config={"radius": 0.3, "fill_color": WHITE, "fill_opacity": 0.9},
            edge_config={"stroke_width": 2},
        ).scale(1.2)

        self.play(Create(g), run_time=1.5)
        self.next_slide()

        # Goal text
        goal_text = Text(
            "Goal: Color vertices so no adjacent vertices share a color",
            font_size=24,
        ).to_edge(DOWN, buff=1.0)
        self.play(FadeIn(goal_text))
        self.next_slide()

        # Show invalid coloring (conflict)
        invalid_label = Text("Invalid coloring:", font_size=24, color=RED).shift(
            RIGHT * 4 + UP * 2
        )
        self.play(FadeIn(invalid_label))

        # Color vertices 0 and 1 the same (they share an edge)
        self.play(
            g.vertices[0].animate.set_fill(RED, opacity=1),
            g.vertices[1].animate.set_fill(RED, opacity=1),
        )
        self.next_slide()

        # Show conflict indicator on edge (0,1)
        edge_01 = g.edges[(0, 1)]
        conflict_highlight = edge_01.copy().set_stroke(YELLOW, width=8)
        conflict_x = Cross(scale_factor=0.3, color=YELLOW).move_to(edge_01.get_center())

        self.play(Create(conflict_highlight), Create(conflict_x))
        self.next_slide()

        # Remove conflict indicators and show valid coloring
        self.play(FadeOut(conflict_highlight), FadeOut(conflict_x), FadeOut(invalid_label))

        valid_label = Text("Valid coloring:", font_size=24, color=GREEN).shift(
            RIGHT * 4 + UP * 2
        )
        self.play(FadeIn(valid_label))

        # Apply valid 3-coloring: 0=Red, 1=Blue, 2=Green, 3=Red, 4=Blue
        colors = {0: RED, 1: BLUE, 2: GREEN, 3: RED, 4: BLUE}
        animations = [
            g.vertices[v].animate.set_fill(c, opacity=1) for v, c in colors.items()
        ]
        self.play(*animations)
        self.next_slide()

        # Show chromatic number
        chi_text = MathTex(r"\chi(G) = 3", font_size=48).shift(RIGHT * 4)
        chi_label = Text("(chromatic number)", font_size=20, color=GRAY_A).next_to(
            chi_text, DOWN
        )
        self.play(Write(chi_text), FadeIn(chi_label))
        self.next_slide()
