"""Slide 9: Multi-Sample Extraction from Dirac."""

from manim import *
from manim_slides import Slide


class DiracExtractionSlide(Slide):
    """Shows the multi-sample, multi-threshold extraction process."""

    def construct(self):
        # Title
        title = Text("Extracting Independent Sets", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.next_slide()

        # Solution vector example
        sol_title = Text("Dirac solution vector:", font_size=24).shift(UP * 2 + LEFT * 3)

        # Create a visual solution vector
        sol_values = [0.31, 0.02, 0.28, 0.05, 0.34]
        sol_entries = VGroup()
        for i, val in enumerate(sol_values):
            entry = VGroup(
                MathTex(f"x_{i}", font_size=20, color=GRAY_A),
                MathTex(f"= {val:.2f}", font_size=24),
            ).arrange(RIGHT, buff=0.1)
            sol_entries.add(entry)
        sol_entries.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        sol_entries.next_to(sol_title, DOWN, buff=0.3, aligned_edge=LEFT)

        self.play(FadeIn(sol_title), FadeIn(sol_entries))
        self.next_slide()

        # Threshold-based extraction
        thresh_title = Text("Multi-threshold extraction:", font_size=24).shift(UP * 2 + RIGHT * 2)

        thresholds = [0.005, 0.01, 0.05, 0.1, 0.2]
        thresh_entries = VGroup()
        for t in thresholds:
            entry = MathTex(f"\\tau = {t}", font_size=22)
            thresh_entries.add(entry)
        thresh_entries.arrange(DOWN, buff=0.1).next_to(thresh_title, DOWN, buff=0.3)

        self.play(FadeIn(thresh_title), FadeIn(thresh_entries))
        self.next_slide()

        # Highlight different thresholds selecting different vertices
        thresh_explain = Text(
            "Different thresholds yield different independent sets",
            font_size=20,
            color=YELLOW,
        ).shift(UP * 0.5)

        self.play(FadeIn(thresh_explain))
        self.next_slide()

        # Pipeline diagram
        pipeline_y = -1.2

        # Step boxes
        step1 = VGroup(
            Rectangle(width=2.2, height=1, color=BLUE_C, fill_opacity=0.2),
            Text("Support\nSet", font_size=16),
        )
        step1[1].move_to(step1[0])
        step1.shift(LEFT * 4.5 + DOWN * abs(pipeline_y))

        step2 = VGroup(
            Rectangle(width=2.2, height=1, color=GREEN_C, fill_opacity=0.2),
            Text("Greedy\nPrune", font_size=16),
        )
        step2[1].move_to(step2[0])
        step2.shift(LEFT * 1.5 + DOWN * abs(pipeline_y))

        step3 = VGroup(
            Rectangle(width=2.2, height=1, color=PURPLE_C, fill_opacity=0.2),
            Text("Local\nSearch", font_size=16),
        )
        step3[1].move_to(step3[0])
        step3.shift(RIGHT * 1.5 + DOWN * abs(pipeline_y))

        step4 = VGroup(
            Rectangle(width=2.2, height=1, color=YELLOW, fill_opacity=0.2),
            Text("Valid\nIS", font_size=16),
        )
        step4[1].move_to(step4[0])
        step4.shift(RIGHT * 4.5 + DOWN * abs(pipeline_y))

        # Arrows
        arr1 = Arrow(step1[0].get_right(), step2[0].get_left(), buff=0.1, color=WHITE)
        arr2 = Arrow(step2[0].get_right(), step3[0].get_left(), buff=0.1, color=WHITE)
        arr3 = Arrow(step3[0].get_right(), step4[0].get_left(), buff=0.1, color=WHITE)

        pipeline = VGroup(step1, step2, step3, step4, arr1, arr2, arr3)

        self.play(Create(pipeline), run_time=1.5)
        self.next_slide()

        # Result
        result_box = Rectangle(width=8, height=1.2, color=GREEN, fill_opacity=0.1)
        result_box.shift(DOWN * 2.8)

        result_text = VGroup(
            Text("Result:", font_size=22, weight=BOLD, color=GREEN),
            Text("20-30 unique columns per Dirac call", font_size=24),
        ).arrange(RIGHT, buff=0.3).move_to(result_box)

        self.play(Create(result_box), FadeIn(result_text))
        self.next_slide()
