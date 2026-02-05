"""Slide 11: Summary."""

from manim import *
from manim_slides import Slide


class SummarySlide(Slide):
    """Final summary slide with key takeaways."""

    def construct(self):
        # Title
        title = Text("Summary", font_size=44).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.next_slide()

        # Key points
        points = [
            "1. Column Generation decomposes graph coloring into RMP + PSP",
            "2. PSP (MWIS) is the bottleneck - it's NP-hard",
            "3. Classical MILP: exact but slow (1 column/call)",
            "4. Dirac-3: extracts 20-30 columns per call via Motzkin-Straus QP",
            "5. Result: Fewer iterations, better solutions",
        ]

        point_mobjects = VGroup()
        for i, p in enumerate(points):
            color = YELLOW if i == 4 else WHITE
            t = Text(p, font_size=24, color=color)
            point_mobjects.add(t)

        point_mobjects.arrange(DOWN, buff=0.4, aligned_edge=LEFT)
        point_mobjects.shift(UP * 0.5 + LEFT * 0.5)

        for pm in point_mobjects:
            self.play(FadeIn(pm, shift=RIGHT * 0.3), run_time=0.5)
            self.next_slide()

        # Scorecard
        score_title = Text("Dirac vs Greedy Scorecard:", font_size=24, weight=BOLD)
        score_title.shift(DOWN * 2)

        scorecard = VGroup(
            Text("4 wins", font_size=28, color=GREEN),
            Text(" | ", font_size=28),
            Text("4 ties", font_size=28, color=YELLOW),
            Text(" | ", font_size=28),
            Text("0 losses", font_size=28, color=RED),
        ).arrange(RIGHT, buff=0.1)
        scorecard.next_to(score_title, DOWN, buff=0.3)

        self.play(FadeIn(score_title), FadeIn(scorecard))
        self.next_slide()

        # Final message
        final = Text(
            "Quantum advantage in combinatorial optimization",
            font_size=26,
            color=PURPLE_A,
            weight=BOLD,
        ).to_edge(DOWN, buff=0.4)

        self.play(FadeIn(final))
        self.next_slide()
