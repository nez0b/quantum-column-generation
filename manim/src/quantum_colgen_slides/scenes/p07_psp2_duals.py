"""PSP#2 duals reveal — mid CG iter, active set is a filtered subgraph."""

import sys
from pathlib import Path

from manim import (
    Scene, Tex, FadeIn, FadeOut, Write, VGroup,
    UP, DOWN, LEFT, RIGHT,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from components import theme  # noqa: F401
from components.theme import FG_PRIMARY, FG_SECONDARY
from components.psp_loader import load_psp, PSP2_PATH
from components.psp_graph import build_psp_graph, dim_inactive_anims
from components.qci_colors import QCI_TEAL


class PSP2DualsScene(Scene):
    def construct(self):
        psp = load_psp(PSP2_PATH)

        title = Tex(
            r"\textbf{PSP \#2: mid CG iter (differentiated duals)}",
            font_size=36, color=FG_PRIMARY,
        ).to_edge(UP, buff=0.4)
        self.play(Write(title), run_time=0.8)

        g = build_psp_graph(psp, scale=2.2).shift(DOWN * 0.2 + LEFT * 1.4)
        self.play(FadeIn(g, lag_ratio=0.02), run_time=1.2)

        self.play(*dim_inactive_anims(g, psp.active_nodes, active_color=QCI_TEAL, inactive_opacity=0.55),
                  run_time=1.0)

        panel = VGroup(
            Tex(r"\textbf{Pricing subproblem}", font_size=34, color=FG_PRIMARY),
            Tex(rf"active: {len(psp.active_nodes)} / {psp.n}", font_size=30, color=QCI_TEAL),
            Tex(rf"filtered out: {psp.n - len(psp.active_nodes)}", font_size=28, color=FG_SECONDARY),
            Tex(r"$w_v = 1.0$ \text{ (on active)}", font_size=32, color=QCI_TEAL),
            Tex(r"inactive vertices", font_size=24, color=FG_SECONDARY),
            Tex(r"already fractionally covered", font_size=24, color=FG_SECONDARY),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).shift(RIGHT * 3.4)

        self.play(FadeIn(panel, shift=LEFT * 0.3), run_time=0.8)

        self.wait(2.5)
        self.play(FadeOut(title), FadeOut(panel), FadeOut(g), run_time=0.8)
