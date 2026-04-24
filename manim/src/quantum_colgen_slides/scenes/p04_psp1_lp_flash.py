"""PSP#1 — flash all 16 LP columns on the shared graph."""

import sys
from pathlib import Path

from manim import Scene, Tex, FadeIn, UP, DOWN, LEFT

sys.path.insert(0, str(Path(__file__).parent.parent))
from components import theme  # noqa: F401
from components.theme import FG_MUTED
from components.psp_loader import load_psp, PSP1_PATH
from components.psp_graph import build_psp_graph, dim_inactive_anims
from components.column_flash import flash_columns
from components.qci_colors import QCI_TEAL, LP_COLOR


class PSP1LPFlashScene(Scene):
    def construct(self):
        psp = load_psp(PSP1_PATH)

        banner = Tex(
            r"\textbf{PSP \#1 --- LP pricing columns}",
            font_size=40, color=LP_COLOR,
        ).to_edge(UP, buff=0.35)
        self.play(FadeIn(banner), run_time=0.5)

        g = build_psp_graph(psp, scale=2.0).shift(UP * 0.3 + LEFT * 2.2)
        self.play(FadeIn(g, lag_ratio=0.02), run_time=1.0)
        self.play(*dim_inactive_anims(g, psp.active_nodes, active_color=QCI_TEAL), run_time=0.4)

        counter_position = [4.2, 1.8, 0]
        pool_anchor = [-6.1, -2.6, 0]

        pool_label = Tex(r"\textit{pool:}", font_size=26, color=FG_MUTED).move_to([-6.6, -2.3, 0])
        self.play(FadeIn(pool_label), run_time=0.3)

        flash_columns(
            self, g,
            columns=psp.lp_columns,
            oracle_label="LP",
            oracle_color=LP_COLOR,
            counter_position=counter_position,
            pool_anchor=pool_anchor,
            dwell=0.55, hold=0.2, fade_time=0.18,
            pool_cols_per_row=8, pool_row_spacing=0.45,
        )
        self.wait(1.0)
