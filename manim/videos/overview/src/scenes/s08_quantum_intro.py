"""Slide 8: Motzkin-Straus QP and Quantum Approach."""

from manim import *
from manim_slides import Slide


class QuantumIntroSlide(Slide):
    """Introduces the Motzkin-Straus QP formulation and how Dirac solves it."""

    def construct(self):
        # Title
        title = Text("The Quantum Approach", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.next_slide()

        # Motzkin-Straus theorem
        ms_title = Text(
            "Motzkin-Straus Theorem",
            font_size=28,
            weight=BOLD,
            color=PURPLE_A,
        ).shift(UP * 1.8)

        ms_formula = MathTex(
            r"\min_{\mathbf{x} \in \Delta} \mathbf{x}^T A_{\bar{G}} \mathbf{x}",
            font_size=40,
        ).next_to(ms_title, DOWN, buff=0.4)

        where_text = Text("where:", font_size=20, color=GRAY_A).next_to(
            ms_formula, DOWN, buff=0.3, aligned_edge=LEFT
        )

        explanations = VGroup(
            MathTex(r"\bar{G} = \text{complement graph}", font_size=24),
            MathTex(r"A_{\bar{G}} = \text{adjacency matrix of } \bar{G}", font_size=24),
            MathTex(r"\Delta = \text{probability simplex}", font_size=24),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT).next_to(where_text, DOWN, buff=0.2, aligned_edge=LEFT)

        self.play(Write(ms_title))
        self.play(Write(ms_formula))
        self.next_slide()

        self.play(FadeIn(where_text), FadeIn(explanations))
        self.next_slide()

        # Dirac connection
        dirac_box = Rectangle(width=10, height=2.5, color=GREEN, fill_opacity=0.1)
        dirac_box.shift(DOWN * 1.8)

        dirac_title = Text("Dirac-3 Entropy Computing", font_size=24, weight=BOLD, color=GREEN)
        dirac_title.move_to(dirac_box.get_top() + DOWN * 0.4)

        dirac_points = VGroup(
            Text("Native QP solver - solves Motzkin-Straus directly", font_size=20),
            Text("Returns continuous solution vectors x", font_size=20),
            Text("Multiple samples per call reveal diverse IS", font_size=20),
        ).arrange(DOWN, buff=0.15, aligned_edge=LEFT).move_to(dirac_box.get_center() + DOWN * 0.2)

        self.play(Create(dirac_box), Write(dirac_title))
        self.play(FadeIn(dirac_points))
        self.next_slide()

        # Flow diagram
        flow_arrow = Arrow(
            dirac_box.get_bottom() + DOWN * 0.2,
            dirac_box.get_bottom() + DOWN * 1.2,
            color=YELLOW,
        )
        extraction_text = Text(
            "Multi-sample extraction",
            font_size=20,
            color=YELLOW,
        ).next_to(flow_arrow, RIGHT, buff=0.2)

        self.play(Create(flow_arrow), FadeIn(extraction_text))
        self.next_slide()
