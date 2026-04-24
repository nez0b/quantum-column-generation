"""PSP#1 — side-by-side bar comparison of LP vs Dirac column pools."""

import sys
from pathlib import Path
from statistics import mean

from manim import (
    Scene, Tex, FadeIn, FadeOut, VGroup,
    UP, DOWN, RIGHT,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from components import theme  # noqa: F401
from components.theme import FG_PRIMARY
from components.psp_loader import load_psp, PSP1_PATH
from components.compare_bars import bar_pair, legend as legend_group


class PSP1CompareScene(Scene):
    def construct(self):
        psp = load_psp(PSP1_PATH)
        lp_count = psp.lp_stats["count"]
        dirac_count = psp.dirac_stats["count"]
        lp_avg_size = psp.lp_stats["avg_size"]
        dirac_avg_size = psp.dirac_stats["avg_size"]
        lp_avg_prof = mean(psp.lp_stats["dual_sums"])
        dirac_avg_prof = mean(psp.dirac_stats["dual_sums"])

        title = Tex(
            r"\textbf{PSP \#1 --- oracle comparison}",
            font_size=40, color=FG_PRIMARY,
        ).to_edge(UP, buff=0.4)
        self.play(FadeIn(title), run_time=0.5)

        count_pair = bar_pair("columns per call", lp_count, dirac_count, max(lp_count, dirac_count) * 1.15)
        size_pair = bar_pair("avg IS size", lp_avg_size, dirac_avg_size, max(lp_avg_size, dirac_avg_size) * 1.15)
        prof_pair = bar_pair(r"avg profit $\Sigma w$", lp_avg_prof, dirac_avg_prof, max(lp_avg_prof, dirac_avg_prof) * 1.15)

        group = VGroup(count_pair, size_pair, prof_pair).arrange(RIGHT, buff=0.9).shift(DOWN * 0.3)
        self.play(FadeIn(group, lag_ratio=0.2), run_time=1.5)

        legend = legend_group().to_edge(DOWN, buff=0.5)
        self.play(FadeIn(legend), run_time=0.5)

        self.wait(3.0)
        self.play(FadeOut(title), FadeOut(group), FadeOut(legend), run_time=0.8)
