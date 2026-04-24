"""Slide 1: Title slide."""

from manim import *
from manim_slides import Slide


class TitleSlide(Slide):
    """Title slide for the presentation."""

    def construct(self):
        # Main title
        title = Text(
            "Quantum Column Generation",
            font_size=56,
            weight=BOLD,
        ).shift(UP * 0.5)

        subtitle = Text(
            "for Graph Coloring",
            font_size=44,
            color=GRAY_A,
        ).next_to(title, DOWN, buff=0.3)

        # Technology line
        tech_line = Text(
            "Using QCi Dirac-3 Entropy Computing",
            font_size=28,
            color=BLUE_C,
        ).next_to(subtitle, DOWN, buff=0.8)

        # Reference
        reference = Text(
            "Based on arXiv:2301.02637v2",
            font_size=20,
            color=GRAY_B,
        ).to_edge(DOWN, buff=0.5)

        # Animate title
        self.play(Write(title), run_time=1.5)
        self.play(FadeIn(subtitle, shift=UP * 0.3), run_time=1.0)
        self.next_slide()

        # Show tech and reference
        self.play(FadeIn(tech_line), FadeIn(reference), run_time=1.0)
        self.next_slide()
