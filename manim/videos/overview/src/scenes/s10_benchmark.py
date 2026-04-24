"""Slide 10: Benchmark Results - Hero Slide."""

from manim import *
from manim_slides import Slide


class BenchmarkSlide(Slide):
    """Shows benchmark results comparing Dirac to classical methods."""

    def construct(self):
        # Title
        title = Text("Benchmark Results", font_size=40).to_edge(UP, buff=0.5)
        self.play(Write(title))
        self.next_slide()

        # ER(40,0.3) - Dirac wins by 2 colors
        graph1_title = Text(
            "ER(40, 0.3) - 40 nodes, 244 edges",
            font_size=24,
            weight=BOLD,
        ).shift(UP * 1.8 + LEFT * 2.5)

        # Table headers
        headers = ["Method", r"\chi", "Cols/Call", "Iterations"]
        header_row = VGroup()
        x_positions = [-2.5, -0.5, 1.5, 3.5]
        for i, h in enumerate(headers):
            if h == r"\chi":
                t = MathTex(h, font_size=24, color=GRAY_A)
            else:
                t = Text(h, font_size=18, weight=BOLD, color=GRAY_A)
            t.move_to([x_positions[i], 1.2, 0])
            header_row.add(t)

        self.play(FadeIn(graph1_title), FadeIn(header_row))
        self.next_slide()

        # Data rows for ER(40,0.3)
        data = [
            ("Greedy", "8", "-", "-", WHITE),
            ("MILP", "8", "1.0", "130", WHITE),
            ("LP", "8", "10.2", "19", WHITE),
            ("Dirac", "6", "20.8", "13", GREEN),
        ]

        rows = VGroup()
        for i, (method, chi, cpc, iters, color) in enumerate(data):
            y_pos = 0.7 - i * 0.5
            row = VGroup(
                Text(method, font_size=18, color=color).move_to([x_positions[0], y_pos, 0]),
                Text(chi, font_size=18, color=color, weight=BOLD if method == "Dirac" else NORMAL).move_to([x_positions[1], y_pos, 0]),
                Text(cpc, font_size=18).move_to([x_positions[2], y_pos, 0]),
                Text(iters, font_size=18).move_to([x_positions[3], y_pos, 0]),
            )
            rows.add(row)

        for row in rows:
            self.play(FadeIn(row), run_time=0.3)
        self.next_slide()

        # Highlight Dirac win
        win_box = SurroundingRectangle(rows[3], color=GREEN, buff=0.1)
        win_label = Text("2 colors better!", font_size=20, color=GREEN).next_to(
            win_box, RIGHT, buff=0.3
        )
        self.play(Create(win_box), FadeIn(win_label))
        self.next_slide()

        # Move first table up and add second table
        first_section = VGroup(graph1_title, header_row, rows, win_box, win_label)
        self.play(first_section.animate.shift(UP * 1.5).scale(0.8))

        # ER(50,0.3) - Dirac matches greedy
        graph2_title = Text(
            "ER(50, 0.3) - 50 nodes, 368 edges",
            font_size=22,
            weight=BOLD,
        ).shift(DOWN * 0.5 + LEFT * 2.5)

        # Second table headers
        header_row2 = VGroup()
        for i, h in enumerate(headers):
            if h == r"\chi":
                t = MathTex(h, font_size=22, color=GRAY_A)
            else:
                t = Text(h, font_size=16, weight=BOLD, color=GRAY_A)
            t.move_to([x_positions[i], -1.0, 0])
            header_row2.add(t)

        self.play(FadeIn(graph2_title), FadeIn(header_row2))

        # Data rows for ER(50,0.3)
        data2 = [
            ("Greedy", "8", "-", "-", YELLOW),
            ("MILP", "11", "1.0", "189", RED),
            ("LP", "10", "8.2", "39", ORANGE),
            ("Dirac", "8", "29.7", "18", GREEN),
        ]

        rows2 = VGroup()
        for i, (method, chi, cpc, iters, color) in enumerate(data2):
            y_pos = -1.5 - i * 0.45
            row = VGroup(
                Text(method, font_size=16, color=color).move_to([x_positions[0], y_pos, 0]),
                Text(chi, font_size=16, color=color, weight=BOLD if method in ["Dirac", "Greedy"] else NORMAL).move_to([x_positions[1], y_pos, 0]),
                Text(cpc, font_size=16).move_to([x_positions[2], y_pos, 0]),
                Text(iters, font_size=16).move_to([x_positions[3], y_pos, 0]),
            )
            rows2.add(row)

        for row in rows2:
            self.play(FadeIn(row), run_time=0.2)
        self.next_slide()

        # Dirac matches greedy highlight
        tie_label = Text(
            "Dirac matches greedy with 10x fewer iterations!",
            font_size=18,
            color=GREEN,
        ).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(tie_label))
        self.next_slide()
