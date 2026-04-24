"""PSP#2 — flash 6 Dirac columns."""

import sys
from pathlib import Path

from manim import Scene, Tex, FadeIn, UP, LEFT

sys.path.insert(0, str(Path(__file__).parent.parent))
from components import theme  # noqa: F401
from components.theme import FG_MUTED
from components.psp_loader import load_psp, PSP2_PATH
from components.psp_graph import build_psp_graph, dim_inactive_anims
from components.column_flash import flash_columns
from components.qci_colors import QCI_TEAL, DIRAC_COLOR


class PSP2DiracFlashScene(Scene):
    def construct(self):
        psp = load_psp(PSP2_PATH)
        banner = Tex(
            r"\textbf{PSP \#2 --- Dirac-3 pricing columns}",
            font_size=40, color=DIRAC_COLOR,
        ).to_edge(UP, buff=0.35)
        self.play(FadeIn(banner), run_time=0.5)

        g = build_psp_graph(psp, scale=2.0).shift(UP * 0.3 + LEFT * 2.2)
        self.play(FadeIn(g, lag_ratio=0.02), run_time=1.0)
        self.play(*dim_inactive_anims(g, psp.active_nodes, active_color=QCI_TEAL, inactive_opacity=0.55),
                  run_time=0.4)

        counter_position = [4.2, 1.8, 0]
        pool_anchor = [-6.1, -2.6, 0]
        self.play(FadeIn(Tex(r"\textit{pool:}", font_size=26, color=FG_MUTED).move_to([-6.6, -2.3, 0])), run_time=0.3)

        flash_columns(
            self, g,
            columns=psp.dirac_columns,
            oracle_label="Dirac",
            oracle_color=DIRAC_COLOR,
            counter_position=counter_position,
            pool_anchor=pool_anchor,
            dwell=1.3, hold=0.5, fade_time=0.3,
            pool_cols_per_row=8, pool_row_spacing=0.45,
        )
        self.wait(1.2)
