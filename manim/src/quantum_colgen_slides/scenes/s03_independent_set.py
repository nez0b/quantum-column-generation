"""Slide 3: Column = Independent Set."""

from manim import *
from manim_slides import Slide


class IndependentSetSlide(Slide):
    """Shows how independent sets become columns in the constraint matrix."""

    def construct(self):
        # Title
        title = Text("Column = Independent Set", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.next_slide()

        # Create graph (same as slide 2)
        vertices = [0, 1, 2, 3, 4]
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]

        g = Graph(
            vertices,
            edges,
            layout="kamada_kawai",
            labels=True,
            vertex_config={"radius": 0.3, "fill_color": GRAY, "fill_opacity": 0.5},
            edge_config={"stroke_width": 2},
        ).shift(LEFT * 3)

        self.play(Create(g), run_time=1.0)
        self.next_slide()

        # Definition
        is_def = Text(
            "Independent Set: No two vertices share an edge",
            font_size=22,
        ).to_edge(DOWN, buff=0.8)
        self.play(FadeIn(is_def))
        self.next_slide()

        # Highlight IS {0, 3}
        is_label = MathTex(r"S_1 = \{0, 3\}", font_size=36, color=YELLOW).shift(
            LEFT * 3 + DOWN * 2.5
        )
        self.play(
            g.vertices[0].animate.set_fill(YELLOW, opacity=1).scale(1.2),
            g.vertices[3].animate.set_fill(YELLOW, opacity=1).scale(1.2),
            FadeIn(is_label),
        )
        self.next_slide()

        # Show arrow to column vector
        arrow = Arrow(LEFT * 1, RIGHT * 0.5, color=WHITE)
        becomes_text = Text("becomes", font_size=20).next_to(arrow, UP, buff=0.1)
        self.play(Create(arrow), FadeIn(becomes_text))
        self.next_slide()

        # Create column vector for {0, 3} using a full matrix display
        # Use Matrix mobject for proper display
        col1_matrix = Matrix(
            [["1"], ["0"], ["0"], ["1"], ["0"]],
            left_bracket="[",
            right_bracket="]",
            element_to_mobject_config={"font_size": 28},
        ).shift(RIGHT * 2)

        # Color the 1s yellow
        col1_matrix.get_entries()[0].set_color(YELLOW)
        col1_matrix.get_entries()[3].set_color(YELLOW)

        # Vertex labels
        v_labels = VGroup(
            *[
                MathTex(f"v_{i}", font_size=22, color=GRAY_A)
                for i in range(5)
            ]
        ).arrange(DOWN, buff=0.32)
        v_labels.next_to(col1_matrix, LEFT, buff=0.5)

        self.play(Create(col1_matrix), FadeIn(v_labels))
        self.next_slide()

        # Show second IS {1, 4}
        is_label2 = MathTex(r"S_2 = \{1, 4\}", font_size=36, color=BLUE).shift(
            LEFT * 3 + DOWN * 2.5
        )

        # Reset first IS
        self.play(
            g.vertices[0].animate.set_fill(GRAY, opacity=0.5).scale(1 / 1.2),
            g.vertices[3].animate.set_fill(GRAY, opacity=0.5).scale(1 / 1.2),
            ReplacementTransform(is_label, is_label2),
        )

        # Highlight second IS
        self.play(
            g.vertices[1].animate.set_fill(BLUE, opacity=1).scale(1.2),
            g.vertices[4].animate.set_fill(BLUE, opacity=1).scale(1.2),
        )
        self.next_slide()

        # Show second column
        col2_matrix = Matrix(
            [["0"], ["1"], ["0"], ["0"], ["1"]],
            left_bracket="[",
            right_bracket="]",
            element_to_mobject_config={"font_size": 28},
        ).shift(RIGHT * 4.5)

        # Color the 1s blue
        col2_matrix.get_entries()[1].set_color(BLUE)
        col2_matrix.get_entries()[4].set_color(BLUE)

        self.play(Create(col2_matrix))
        self.next_slide()

        # Show matrix concept
        matrix_text = Text(
            "Constraint matrix A: each column is an IS",
            font_size=24,
        ).to_edge(UP, buff=1.5)
        self.play(FadeIn(matrix_text))
        self.next_slide()
