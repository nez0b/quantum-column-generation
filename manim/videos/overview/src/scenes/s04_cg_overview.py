"""Slide 4: Column Generation Overview."""

from manim import *
from manim_slides import Slide


class CGOverviewSlide(Slide):
    """Shows the CG loop: RMP <-> PSP."""

    def construct(self):
        # Title
        title = Text("Column Generation Algorithm", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.next_slide()

        # RMP box
        rmp = Rectangle(width=4, height=2, color=BLUE_C, fill_opacity=0.2)
        rmp_title = Text("RMP", font_size=32, weight=BOLD).move_to(rmp.get_top() + DOWN * 0.4)
        rmp_desc = Text(
            "Restricted\nMaster Problem",
            font_size=18,
            color=GRAY_A,
        ).move_to(rmp.get_center() + DOWN * 0.2)
        rmp_group = VGroup(rmp, rmp_title, rmp_desc).shift(LEFT * 3.5)

        # PSP box
        psp = Rectangle(width=4, height=2, color=GREEN_C, fill_opacity=0.2)
        psp_title = Text("PSP", font_size=32, weight=BOLD).move_to(psp.get_top() + DOWN * 0.4)
        psp_desc = Text(
            "Pricing\nSubproblem",
            font_size=18,
            color=GRAY_A,
        ).move_to(psp.get_center() + DOWN * 0.2)
        psp_group = VGroup(psp, psp_title, psp_desc).shift(RIGHT * 3.5)

        self.play(Create(rmp_group), run_time=1.0)
        self.play(Create(psp_group), run_time=1.0)
        self.next_slide()

        # Arrow: RMP -> PSP (dual variables)
        arr_top = Arrow(
            rmp.get_right() + UP * 0.3,
            psp.get_left() + UP * 0.3,
            color=YELLOW,
            buff=0.1,
        )
        arr_top_label = MathTex(r"\lambda_v", font_size=28, color=YELLOW).next_to(
            arr_top, UP, buff=0.1
        )
        dual_desc = Text("dual variables", font_size=16, color=GRAY_A).next_to(
            arr_top_label, UP, buff=0.05
        )

        self.play(Create(arr_top), Write(arr_top_label), FadeIn(dual_desc))
        self.next_slide()

        # Arrow: PSP -> RMP (new columns)
        arr_bot = Arrow(
            psp.get_left() + DOWN * 0.3,
            rmp.get_right() + DOWN * 0.3,
            color=PURPLE,
            buff=0.1,
        )
        arr_bot_label = Text("new columns", font_size=18, color=PURPLE).next_to(
            arr_bot, DOWN, buff=0.1
        )

        self.play(Create(arr_bot), Write(arr_bot_label))
        self.next_slide()

        # Iteration loop indicator
        loop_text = Text(
            "Iterate until no profitable columns",
            font_size=22,
            color=GRAY_A,
        ).to_edge(DOWN, buff=1.0)

        # Add circular arrow around the diagram
        loop_path = CurvedArrow(
            psp.get_bottom() + DOWN * 0.5 + LEFT * 1,
            rmp.get_bottom() + DOWN * 0.5 + RIGHT * 1,
            angle=-TAU / 3,
            color=WHITE,
        )

        self.play(Create(loop_path), FadeIn(loop_text))
        self.next_slide()

        # Key insight
        insight = VGroup(
            Text("Key insight:", font_size=24, weight=BOLD, color=YELLOW),
            Text(
                "More columns per PSP call = faster convergence",
                font_size=22,
            ),
        ).arrange(DOWN, buff=0.1).to_edge(DOWN, buff=0.3)

        self.play(FadeOut(loop_text), FadeIn(insight))
        self.next_slide()
