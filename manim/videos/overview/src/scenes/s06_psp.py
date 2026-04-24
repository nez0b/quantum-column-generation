"""Slide 6: Pricing Subproblem (PSP)."""

from manim import *
from manim_slides import Slide


class PSPSlide(Slide):
    """Explains the Pricing Subproblem: Maximum Weight Independent Set."""

    def construct(self):
        # Title
        title = Text("Pricing Subproblem", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.next_slide()

        # MWIS formulation
        mwis_title = Text("Maximum Weight Independent Set:", font_size=24, color=GREEN_C)
        mwis_title.shift(UP * 1.5)

        objective = MathTex(
            r"\max \sum_{v \in V} \lambda_v \cdot x_v",
            font_size=40,
        ).next_to(mwis_title, DOWN, buff=0.5)

        constraint = MathTex(
            r"\text{s.t.} \quad x_u + x_v \leq 1 \quad \forall (u, v) \in E",
            font_size=32,
        ).next_to(objective, DOWN, buff=0.3)

        binary = MathTex(
            r"x_v \in \{0, 1\}",
            font_size=32,
        ).next_to(constraint, DOWN, buff=0.3)

        self.play(Write(mwis_title))
        self.play(Write(objective))
        self.next_slide()

        self.play(Write(constraint))
        self.play(Write(binary))
        self.next_slide()

        # Profitability condition
        profit_box = Rectangle(width=7, height=1.5, color=YELLOW, fill_opacity=0.1)
        profit_box.shift(DOWN * 1.5)

        profit_text = MathTex(
            r"\text{Profitable if: } \sum_{v \in S} \lambda_v > 1",
            font_size=32,
            color=YELLOW,
        ).move_to(profit_box)

        self.play(Create(profit_box), Write(profit_text))
        self.next_slide()

        # NP-hard warning
        warning_box = Rectangle(width=4, height=1.2, color=RED, fill_opacity=0.3)
        warning_box.shift(DOWN * 3)

        warning_icon = Text("!", font_size=48, weight=BOLD, color=RED)
        warning_text = Text("NP-hard", font_size=28, weight=BOLD, color=RED)
        warning_group = VGroup(warning_icon, warning_text).arrange(RIGHT, buff=0.3)
        warning_group.move_to(warning_box)

        self.play(Create(warning_box), FadeIn(warning_group))
        self.next_slide()

        # Classical approach limitation hint
        hint = Text(
            "Classical MILP: exact but slow, only 1 column per call",
            font_size=20,
            color=GRAY_A,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(hint))
        self.next_slide()
