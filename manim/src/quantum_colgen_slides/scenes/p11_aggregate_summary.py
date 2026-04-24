"""Aggregate summary — cols/call and CG iterations for n=100 × density.

Numbers pulled from RF-branching/benchmarks.db (seed=0) 2026-04-24:

    n=100  p=0.3  CG 10.97 cols/call  86 iters     QCG 50.49 cols/call  41 iters
    n=100  p=0.5  CG  9.31 cols/call  64 iters     QCG 26.46 cols/call  35 iters
    n=100  p=0.7  CG  9.41 cols/call  39 iters     QCG 18.42 cols/call  26 iters
"""

import sys
from pathlib import Path

from manim import (
    Scene, Tex, FadeIn, FadeOut, VGroup,
    UP, DOWN, RIGHT,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from components import theme  # noqa: F401
from components.theme import FG_PRIMARY, FG_SECONDARY
from components.compare_bars import bar_pair, legend as legend_group


N100_ROWS = [
    (0.3, 10.97, 50.49, 86, 41),
    (0.5,  9.31, 26.46, 64, 35),
    (0.7,  9.41, 18.42, 39, 26),
]


class AggregateSummaryScene(Scene):
    def construct(self):
        title = Tex(
            r"\textbf{Aggregate: $\mathrm{ER}(n{=}100)$ across densities}",
            font_size=40, color=FG_PRIMARY,
        ).to_edge(UP, buff=0.4)
        subtitle = Tex(
            r"cols per call \; (bigger $=$ fewer pricing rounds)",
            font_size=28, color=FG_SECONDARY,
        ).next_to(title, DOWN, buff=0.2)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.6)

        max_cpc = max(max(r[1], r[2]) for r in N100_ROWS) * 1.1
        cpc_bars = VGroup(*[
            bar_pair(rf"$p = {p}$", cg, qcg, max_cpc)
            for (p, cg, qcg, *_rest) in N100_ROWS
        ]).arrange(RIGHT, buff=1.2).shift(UP * 0.3)
        self.play(FadeIn(cpc_bars, lag_ratio=0.25), run_time=1.8)

        legend = legend_group().to_edge(DOWN, buff=1.6)
        self.play(FadeIn(legend), run_time=0.4)

        iter_notes = VGroup()
        for bar_group, (p, cg, qcg, cg_it, qcg_it) in zip(cpc_bars, N100_ROWS):
            note = Tex(
                rf"CG: {cg_it} iters $\to$ QCG: {qcg_it}",
                font_size=26, color=FG_SECONDARY,
            ).next_to(bar_group, DOWN, buff=0.5)
            iter_notes.add(note)
        self.play(FadeIn(iter_notes, shift=UP * 0.1), run_time=0.8)

        self.wait(4.0)
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(cpc_bars),
                  FadeOut(legend), FadeOut(iter_notes), run_time=0.8)
