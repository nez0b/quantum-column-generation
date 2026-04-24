"""PSP demo — closing slate."""

import sys
from pathlib import Path

from manim import Scene, Tex, FadeIn, FadeOut, Write, UP, DOWN

sys.path.insert(0, str(Path(__file__).parent.parent))
from components import theme  # noqa: F401
from components.theme import FG_PRIMARY, FG_MUTED
from components.qci_colors import QCI_ORANGE


class ClosingScene(Scene):
    def construct(self):
        line1 = Tex(
            r"\textbf{Per-call yield $\times$ IS size}",
            font_size=56, color=FG_PRIMARY,
        ).shift(UP * 0.8)
        line2 = Tex(
            r"$\to$ fewer CG iterations at larger $n$",
            font_size=44, color=QCI_ORANGE,
        ).next_to(line1, DOWN, buff=0.35)
        ref = Tex(
            r"\small Full deck: \texttt{RF-branching/slides/qcg\_vs\_cg\_demo/qcg\_vs\_cg\_demo.pdf}",
            font_size=22, color=FG_MUTED,
        ).to_edge(DOWN, buff=0.6)

        self.play(Write(line1), run_time=1.0)
        self.play(FadeIn(line2, shift=UP * 0.2), run_time=0.8)
        self.play(FadeIn(ref), run_time=0.5)
        self.wait(2.2)
        self.play(FadeOut(line1), FadeOut(line2), FadeOut(ref))
