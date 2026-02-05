"""Slide 7: Why Classical MILP Fails at Scale."""

from manim import *
from manim_slides import Slide


class MILPLimitationSlide(Slide):
    """Shows the limitation of classical MILP: only 1 column per call."""

    def construct(self):
        # Title
        title = Text("The MILP Bottleneck", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.next_slide()

        # Comparison setup
        milp_col = VGroup(
            Text("Classical MILP", font_size=28, weight=BOLD, color=RED),
            Text("Exact solver", font_size=20, color=GRAY_A),
        ).arrange(DOWN, buff=0.1).shift(LEFT * 3.5 + UP * 1.5)

        dirac_col = VGroup(
            Text("Dirac-3", font_size=28, weight=BOLD, color=GREEN),
            Text("Quantum annealer", font_size=20, color=GRAY_A),
        ).arrange(DOWN, buff=0.1).shift(RIGHT * 3.5 + UP * 1.5)

        self.play(FadeIn(milp_col), FadeIn(dirac_col))
        self.next_slide()

        # Columns per call comparison
        cpc_label = Text("Columns per call:", font_size=24).shift(UP * 0.3)

        milp_cpc = MathTex(r"1", font_size=72, color=RED).shift(LEFT * 3.5 + DOWN * 0.5)
        dirac_cpc = MathTex(r"20+", font_size=72, color=GREEN).shift(RIGHT * 3.5 + DOWN * 0.5)

        self.play(FadeIn(cpc_label))
        self.play(Write(milp_cpc))
        self.play(Write(dirac_cpc))
        self.next_slide()

        # Iteration comparison
        iter_label = Text("Iterations for ER(50,0.3):", font_size=24).shift(DOWN * 1.5)

        milp_iter = VGroup(
            MathTex(r"189", font_size=48, color=RED),
            Text("iterations", font_size=18),
        ).arrange(DOWN, buff=0.1).shift(LEFT * 3.5 + DOWN * 2.5)

        dirac_iter = VGroup(
            MathTex(r"18", font_size=48, color=GREEN),
            Text("iterations", font_size=18),
        ).arrange(DOWN, buff=0.1).shift(RIGHT * 3.5 + DOWN * 2.5)

        self.play(FadeIn(iter_label))
        self.play(Write(milp_iter), Write(dirac_iter))
        self.next_slide()

        # 10x improvement highlight
        improvement = VGroup(
            Text("10x", font_size=56, weight=BOLD, color=YELLOW),
            Text("fewer iterations", font_size=24),
        ).arrange(DOWN, buff=0.1).move_to(ORIGIN + DOWN * 0.5)

        # Fade out comparison numbers, show improvement
        self.play(
            FadeOut(milp_cpc),
            FadeOut(dirac_cpc),
            FadeOut(cpc_label),
        )
        self.play(FadeIn(improvement))
        self.next_slide()

        # Key insight
        insight = Text(
            "More diverse columns = better final coloring",
            font_size=22,
            color=YELLOW,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(insight))
        self.next_slide()
