"""Full benchmark table — animated 9×7 grid mirroring figures/12_summary_table.tex.

Rows: 9 (n, p) instances from benchmark_aggregate.json.
Cols: Instance | MILP χ | Hexaly χ | CG χ(iters/cpc) | QCG χ(iters/cpc)
      | BP-ISF χ(iters/cpc) | QBP-ISF χ(iters/cpc)

Best-χ-per-row marked (dark green). After all rows arrive, cpc cells where
QCG > CG (or QBP > BP) get a brief orange outline pulse to drive the point home.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from manim import (
    Scene, Tex, FadeIn, FadeOut, VGroup, Rectangle, Line,
    UP, DOWN, LEFT, RIGHT, BOLD,
    Indicate, AnimationGroup,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from components import theme  # noqa: F401
from components.theme import FG_PRIMARY, FG_SECONDARY, FG_MUTED
from components.qci_colors import LP_COLOR, DIRAC_COLOR, QCI_GREEN, QCI_ORANGE
from components.psp_loader import BENCHMARK_PATH


# Column x-positions, tuned so the whole row fits in a 16:9 frame.
COL_X = {
    "instance": -5.8,
    "milp":     -3.4,
    "hexaly":   -2.4,
    "cg":       -0.6,
    "qcg":       1.4,
    "bp":        3.4,
    "qbp":       5.4,
}
ROW_STEP = 0.42
HEADER_Y = 3.05
FIRST_ROW_Y = 1.85


def _chi_cell(chi, is_best, method_color=FG_PRIMARY):
    if chi is None:
        return Tex(r"--", font_size=24, color=FG_MUTED)
    txt = Tex(rf"$\mathbf{{{chi}}}$", font_size=24,
              color=QCI_GREEN if is_best else method_color)
    return txt


def _method_cell(chi, iters, cpc, is_best, method_color):
    """Cell content: chi (iters/cpc), with bold chi + muted parenthetical."""
    if chi is None:
        return Tex(r"--", font_size=22, color=FG_MUTED)
    chi_color = QCI_GREEN if is_best else method_color
    if iters is None or cpc is None:
        return Tex(rf"$\mathbf{{{chi}}}$", font_size=22, color=chi_color)
    # Format: 11 (12/13.9)
    main = Tex(rf"$\mathbf{{{chi}}}$", font_size=22, color=chi_color)
    paren = Tex(rf"({int(iters)}/{cpc:.1f})", font_size=18, color=FG_MUTED)
    paren.next_to(main, RIGHT, buff=0.1)
    return VGroup(main, paren)


class SummaryTableScene(Scene):
    def construct(self):
        data = json.loads(Path(BENCHMARK_PATH).read_text())["instances"]
        # Sort by (n, p) for consistent order
        data = sorted(data, key=lambda x: (x["n"], x["p"]))

        title = Tex(
            r"\textbf{Full benchmark table — 9 ER instances, all methods}",
            font_size=34, color=FG_PRIMARY,
        ).to_edge(UP, buff=0.15)
        self.play(FadeIn(title), run_time=0.5)

        # Headers row.
        hdr_entries = {
            "instance": Tex(r"\textbf{Instance}", font_size=22, color=FG_PRIMARY),
            "milp":     Tex(r"\textbf{MILP}", font_size=22, color=FG_SECONDARY),
            "hexaly":   Tex(r"\textbf{Hexaly}", font_size=22, color=FG_SECONDARY),
            "cg":       Tex(r"\textbf{CG}", font_size=22, color=LP_COLOR),
            "qcg":      Tex(r"\textbf{QCG}", font_size=22, color=DIRAC_COLOR),
            "bp":       Tex(r"\textbf{BP-ISF}", font_size=22, color=LP_COLOR),
            "qbp":      Tex(r"\textbf{QBP-ISF}", font_size=22, color=DIRAC_COLOR),
        }
        for key, mob in hdr_entries.items():
            mob.move_to([COL_X[key], HEADER_Y, 0])
        hdr_group = VGroup(*hdr_entries.values())

        # Subheader: "χ (iters/cpc)"
        subhdr = Tex(r"$\chi$ (iters / cols-per-call)", font_size=18, color=FG_MUTED)
        subhdr.move_to([(COL_X["cg"] + COL_X["qbp"]) / 2, HEADER_Y - 0.45, 0])

        # Separator line between header/subhdr and first row.
        sep = Line(
            start=[COL_X["instance"] - 0.3, HEADER_Y - 0.85, 0],
            end=[COL_X["qbp"] + 0.9, HEADER_Y - 0.85, 0],
            stroke_color=FG_MUTED, stroke_width=1.0,
        )
        self.play(FadeIn(hdr_group), FadeIn(subhdr), FadeIn(sep), run_time=0.6)

        # Rows.
        rows = []           # list of (row_group, cpc_cells_for_pulse)
        pulse_targets = []  # cells to pulse: QCG cpc > CG cpc
        pulse_bp_targets = []  # QBP cpc > BP cpc

        for i, inst in enumerate(data):
            y = FIRST_ROW_Y - i * ROW_STEP
            m = inst["methods"]
            best_chi = inst["best_chi"]

            inst_lbl = Tex(
                rf"$n{{=}}{inst['n']},\; p{{=}}{inst['p']}$",
                font_size=22, color=FG_PRIMARY,
            ).move_to([COL_X["instance"], y, 0], aligned_edge=LEFT)

            milp_cell   = _chi_cell(m["milp"]["chi"],   m["milp"]["chi"]   == best_chi).move_to([COL_X["milp"],   y, 0])
            hexaly_cell = _chi_cell(m["hexaly"]["chi"], m["hexaly"]["chi"] == best_chi).move_to([COL_X["hexaly"], y, 0])

            cg_cell  = _method_cell(m["cg"]["chi"],      m["cg"]["iters"],      m["cg"]["cpc"],      m["cg"]["chi"]      == best_chi, LP_COLOR).move_to([COL_X["cg"],  y, 0])
            qcg_cell = _method_cell(m["qcg"]["chi"],     m["qcg"]["iters"],     m["qcg"]["cpc"],     m["qcg"]["chi"]     == best_chi, DIRAC_COLOR).move_to([COL_X["qcg"], y, 0])
            bp_cell  = _method_cell(m["bp_isf"]["chi"],  m["bp_isf"]["iters"],  m["bp_isf"]["cpc"],  m["bp_isf"]["chi"]  == best_chi, LP_COLOR).move_to([COL_X["bp"],  y, 0])
            qbp_cell = _method_cell(m["qbp_isf"]["chi"], m["qbp_isf"]["iters"], m["qbp_isf"]["cpc"], m["qbp_isf"]["chi"] == best_chi, DIRAC_COLOR).move_to([COL_X["qbp"], y, 0])

            row_group = VGroup(inst_lbl, milp_cell, hexaly_cell, cg_cell, qcg_cell, bp_cell, qbp_cell)
            rows.append(row_group)

            # Track cells to pulse (QCG beats CG, QBP beats BP in cpc).
            if m["qcg"]["cpc"] and m["cg"]["cpc"] and m["qcg"]["cpc"] > m["cg"]["cpc"]:
                pulse_targets.append(qcg_cell)
            if m["qbp_isf"]["cpc"] and m["bp_isf"]["cpc"] and m["qbp_isf"]["cpc"] > m["bp_isf"]["cpc"]:
                pulse_bp_targets.append(qbp_cell)

        # Fade rows in sequentially (fast).
        for r in rows:
            self.play(FadeIn(r, shift=LEFT * 0.15), run_time=0.32)

        self.wait(0.5)

        # Pulse QCG/QBP cpc winners — all at once with orange flash.
        if pulse_targets or pulse_bp_targets:
            self.play(
                *(Indicate(c, color=QCI_ORANGE, scale_factor=1.08) for c in pulse_targets + pulse_bp_targets),
                run_time=1.3,
            )

        self.wait(2.5)

        # Fade everything out.
        self.play(
            FadeOut(title), FadeOut(hdr_group), FadeOut(subhdr), FadeOut(sep),
            *[FadeOut(r) for r in rows],
            run_time=0.8,
        )
