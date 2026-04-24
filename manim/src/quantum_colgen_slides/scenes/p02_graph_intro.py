"""PSP demo — FadeIn the n=20 instance using the cached Kamada-Kawai layout."""

import sys
from pathlib import Path

from manim import (
    Scene, Tex, FadeIn, FadeOut, Create, Write, VGroup,
    UP, DOWN, LEFT, RIGHT,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from components import theme  # noqa: F401
from components.theme import FG_PRIMARY
from components.psp_loader import load_psp, PSP1_PATH
from components.psp_graph import build_psp_graph
from components.qci_colors import QCI_BLUE, QCI_TEAL


class PSPGraphIntroScene(Scene):
    def construct(self):
        psp = load_psp(PSP1_PATH)

        title = Tex(
            r"\textbf{The running instance}", font_size=44, color=FG_PRIMARY,
        ).to_edge(UP, buff=0.5)
        self.play(Write(title), run_time=0.8)

        g = build_psp_graph(psp, scale=2.4).shift(DOWN * 0.4)
        self.play(Create(g, lag_ratio=0.02), run_time=2.5)

        stats = VGroup(
            Tex(r"$|V| = 20$", font_size=36, color=QCI_TEAL),
            Tex(r"$|E| = 131$", font_size=36, color=QCI_TEAL),
            Tex(r"$\chi(G) = 8$", font_size=36, color=QCI_BLUE),
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).to_corner(UP + RIGHT, buff=0.7)
        self.play(FadeIn(stats, shift=LEFT * 0.3), run_time=0.8)

        self.wait(2.0)
        self.play(FadeOut(title), FadeOut(stats), FadeOut(g), run_time=0.8)
