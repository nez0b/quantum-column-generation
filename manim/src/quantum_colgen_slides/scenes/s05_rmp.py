"""Slide 5: Restricted Master Problem (RMP)."""

from manim import *
from manim_slides import Slide


class RMPSlide(Slide):
    """Explains the Restricted Master Problem LP formulation."""

    def construct(self):
        # Title
        title = Text("Restricted Master Problem", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.next_slide()

        # LP formulation
        lp_title = Text("Linear Program:", font_size=24, color=BLUE_C).shift(
            UP * 1.5 + LEFT * 3
        )

        objective = MathTex(
            r"\min \sum_{s \in S} y_s",
            font_size=36,
        ).next_to(lp_title, DOWN, buff=0.5, aligned_edge=LEFT)

        constraint = MathTex(
            r"\text{s.t.} \quad \sum_{s : v \in s} y_s \geq 1 \quad \forall v \in V",
            font_size=32,
        ).next_to(objective, DOWN, buff=0.3, aligned_edge=LEFT)

        nonnegativity = MathTex(
            r"y_s \geq 0 \quad \forall s \in S",
            font_size=32,
        ).next_to(constraint, DOWN, buff=0.3, aligned_edge=LEFT)

        lp_group = VGroup(lp_title, objective, constraint, nonnegativity)

        self.play(Write(lp_title))
        self.play(Write(objective))
        self.next_slide()

        self.play(Write(constraint))
        self.play(Write(nonnegativity))
        self.next_slide()

        # Explanation box
        explain = VGroup(
            Text("Where:", font_size=22, weight=BOLD),
            MathTex(r"S = \text{set of known independent sets}", font_size=24),
            MathTex(r"y_s = \text{selection weight for IS } s", font_size=24),
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT).shift(RIGHT * 2 + UP * 0.5)

        self.play(FadeIn(explain))
        self.next_slide()

        # Dual variables highlight
        dual_box = Rectangle(width=5.5, height=2.2, color=YELLOW, fill_opacity=0.1)
        dual_box.shift(DOWN * 2)

        dual_title = Text("Dual Variables:", font_size=24, weight=BOLD, color=YELLOW)
        dual_title.move_to(dual_box.get_top() + DOWN * 0.3)

        dual_explain = VGroup(
            MathTex(r"\lambda_v = \text{shadow price for covering vertex } v", font_size=22),
            Text("Higher price = vertex harder to cover", font_size=18, color=GRAY_A),
        ).arrange(DOWN, buff=0.15).move_to(dual_box.get_center() + DOWN * 0.2)

        self.play(Create(dual_box), Write(dual_title), FadeIn(dual_explain))
        self.next_slide()

        # Dual variables sent to PSP
        arrow = Arrow(dual_box.get_right(), dual_box.get_right() + RIGHT * 2, color=YELLOW)
        to_psp = Text("To PSP", font_size=20, color=YELLOW).next_to(arrow, UP, buff=0.1)
        self.play(Create(arrow), FadeIn(to_psp))
        self.next_slide()
