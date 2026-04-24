"""PSP demo — title card."""

import sys
from pathlib import Path

from manim import Scene, Tex, FadeIn, FadeOut, Write, UP, DOWN

sys.path.insert(0, str(Path(__file__).parent.parent))
from components import theme  # noqa: F401 — applies light-bg config on import
from components.theme import FG_PRIMARY, FG_SECONDARY
from components.qci_colors import QCI_BLUE, QCI_ORANGE


class PSPTitleScene(Scene):
    def construct(self):
        title = Tex(
            r"\textbf{Column Generation vs Quantum Column Generation}",
            font_size=48, color=FG_PRIMARY,
        ).shift(UP * 0.6)
        tagline = Tex(
            r"\textit{one pricing subproblem at a time}",
            font_size=36, color=FG_SECONDARY,
        ).next_to(title, DOWN, buff=0.4)
        subtitle = Tex(
            r"$\mathrm{ER}(n{=}20,\; p{=}0.7,\; \mathrm{seed}{=}0)$",
            font_size=36, color=QCI_BLUE,
        ).next_to(tagline, DOWN, buff=0.8)
        caption = Tex(
            r"LP pricing \quad vs \quad Dirac-3 pricing",
            font_size=32, color=QCI_ORANGE,
        ).to_edge(DOWN, buff=0.6)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(tagline, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(subtitle), run_time=0.6)
        self.play(FadeIn(caption), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(title), FadeOut(tagline), FadeOut(subtitle), FadeOut(caption))
