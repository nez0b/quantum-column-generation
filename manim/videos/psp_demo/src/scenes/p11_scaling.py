"""Scaling narrative — cpc advantage grows with graph size.

9 instances from benchmark_aggregate.json arranged as a 3-row × 3-col grid:
    rows  = n ∈ {20, 50, 100}  (top → bottom)
    cols  = p ∈ {0.3, 0.5, 0.7}

Each cell = paired bar (CG cpc vs QCG cpc), globally normalized to the
same scale. Animates row-by-row so the widening gap from n=20 → 50 → 100
is the visual payoff. The n=100 row gets a subtle glow to lock in the point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from manim import (
    Scene, Tex, FadeIn, FadeOut, VGroup, Rectangle,
    UP, DOWN, LEFT, RIGHT, BOLD,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from components import theme  # noqa: F401
from components.theme import FG_PRIMARY, FG_SECONDARY
from components.qci_colors import LP_COLOR, DIRAC_COLOR, QCI_ORANGE
from components.psp_loader import BENCHMARK_PATH


# Global bar geometry.
BAR_WIDTH = 0.35
MAX_BAR_HEIGHT = 1.2
HEIGHT_PER_CPC = MAX_BAR_HEIGHT / 55.0  # global scale; max cpc ≈ 50.5


def _cell(cg_cpc: float, qcg_cpc: float) -> VGroup:
    """A bar-pair cell: CG (LP_COLOR) bar and QCG (DIRAC_COLOR) bar, with numeric labels above."""
    cg_h = max(0.04, cg_cpc * HEIGHT_PER_CPC)
    qcg_h = max(0.04, qcg_cpc * HEIGHT_PER_CPC)
    cg_bar = Rectangle(
        width=BAR_WIDTH, height=cg_h,
        fill_color=LP_COLOR, fill_opacity=0.85, stroke_color=LP_COLOR, stroke_width=1.5,
    )
    qcg_bar = Rectangle(
        width=BAR_WIDTH, height=qcg_h,
        fill_color=DIRAC_COLOR, fill_opacity=0.85, stroke_color=DIRAC_COLOR, stroke_width=1.5,
    )
    qcg_bar.next_to(cg_bar, RIGHT, buff=0.08, aligned_edge=DOWN)
    pair = VGroup(cg_bar, qcg_bar)

    cg_num = Tex(f"{cg_cpc:.1f}", font_size=22, color=FG_PRIMARY).next_to(cg_bar, UP, buff=0.05)
    qcg_num = Tex(f"{qcg_cpc:.1f}", font_size=22, color=FG_PRIMARY).next_to(qcg_bar, UP, buff=0.05)
    return VGroup(pair, cg_num, qcg_num)


class ScalingScene(Scene):
    def construct(self):
        data = json.loads(Path(BENCHMARK_PATH).read_text())["instances"]
        # Index by (n, p)
        lookup = {(inst["n"], inst["p"]): inst for inst in data}

        title = Tex(
            r"\textbf{The QCG cols-per-call advantage scales with $n$}",
            font_size=36, color=FG_PRIMARY,
        ).to_edge(UP, buff=0.35)
        subtitle = Tex(
            r"LP vs Dirac-3 cols-per-call on ER($n$, $p$, seed=0)",
            font_size=26, color=FG_SECONDARY,
        ).next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.6)

        # --- grid layout -------------------------------------------------
        ns = [20, 50, 100]
        ps = [0.3, 0.5, 0.7]

        col_x = {0.3: -3.6, 0.5: 0.0, 0.7: 3.6}      # x-position per column (density)
        row_y = {20: 1.35, 50: -0.25, 100: -1.85}    # y-position (row baseline)
        row_color_highlight = {100: QCI_ORANGE}

        # Column headers (density labels).
        col_headers = VGroup()
        for p in ps:
            h = Tex(rf"$p = {p}$", font_size=32, color=FG_PRIMARY)
            h.move_to([col_x[p], 2.35, 0])
            col_headers.add(h)
        self.play(FadeIn(col_headers), run_time=0.4)

        # Row labels (n).
        row_labels = {}
        for n in ns:
            lbl = Tex(rf"\textbf{{$n = {n}$}}", font_size=32, color=FG_PRIMARY)
            lbl.move_to([-6.0, row_y[n], 0])
            row_labels[n] = lbl

        # Pre-position cells (but don't show yet).
        cells = {}
        for n in ns:
            for p in ps:
                m = lookup[(n, p)]["methods"]
                cg_cpc = m["cg"]["cpc"]
                qcg_cpc = m["qcg"]["cpc"]
                cell = _cell(cg_cpc, qcg_cpc)
                cell.move_to([col_x[p], row_y[n], 0], aligned_edge=DOWN)
                cells[(n, p)] = cell

        # Animate row-by-row reveal.
        for n in ns:
            row_cells = VGroup(*[cells[(n, p)] for p in ps])
            anims = [FadeIn(row_labels[n]), FadeIn(row_cells, lag_ratio=0.12)]
            self.play(*anims, run_time=0.9)
            self.wait(0.7)

        # Punctuate the n=100 row: a halo rectangle + brief zoom label.
        n100_row = VGroup(row_labels[100], *[cells[(100, p)] for p in ps])
        halo = Rectangle(
            width=n100_row.width + 0.6, height=n100_row.height + 0.3,
            stroke_color=QCI_ORANGE, stroke_width=2.5,
            fill_color=QCI_ORANGE, fill_opacity=0.05,
        ).move_to(n100_row)
        ratio_hint = Tex(
            r"$\mathrm{QCG}/\mathrm{CG} \approx 4.6\times \; @ \; p{=}0.3$",
            font_size=28, color=QCI_ORANGE,
        ).next_to(n100_row, DOWN, buff=0.3)
        self.play(FadeIn(halo), FadeIn(ratio_hint), run_time=0.8)

        # Legend at the top right.
        legend = VGroup(
            Rectangle(width=0.22, height=0.22, fill_color=LP_COLOR, fill_opacity=0.85, stroke_width=0),
            Tex(r"CG (LP)", font_size=24, color=LP_COLOR),
            Rectangle(width=0.22, height=0.22, fill_color=DIRAC_COLOR, fill_opacity=0.85, stroke_width=0),
            Tex(r"QCG (Dirac-3)", font_size=24, color=DIRAC_COLOR),
        ).arrange(RIGHT, buff=0.2).to_corner(UP + RIGHT, buff=0.7)
        self.play(FadeIn(legend), run_time=0.4)

        self.wait(3.0)
        self.play(
            FadeOut(title), FadeOut(subtitle), FadeOut(col_headers),
            *[FadeOut(lbl) for lbl in row_labels.values()],
            *[FadeOut(c) for c in cells.values()],
            FadeOut(halo), FadeOut(ratio_hint), FadeOut(legend),
            run_time=0.8,
        )
