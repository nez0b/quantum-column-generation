"""QCI brand palette — matches RF-branching/slides/qcg_vs_cg_demo/pipeline/render_figures.py.

Keeps the animation and the beamer deck visually synchronized.
"""

from manim import ManimColor

QCI_BLUE = ManimColor("#004682")
QCI_TEAL = ManimColor("#008080")
QCI_ORANGE = ManimColor("#DC7814")
QCI_GREEN = ManimColor("#288C3C")
QCI_GREY = ManimColor("#B8B8B8")
QCI_DARKGREY = ManimColor("#606060")

LP_COLOR = QCI_BLUE
DIRAC_COLOR = QCI_ORANGE

IS_PALETTE = [
    ManimColor("#004682"),
    ManimColor("#008080"),
    ManimColor("#DC7814"),
    ManimColor("#288C3C"),
    ManimColor("#8E44AD"),
    ManimColor("#C0392B"),
    ManimColor("#16A085"),
    ManimColor("#D4AC0D"),
    ManimColor("#2980B9"),
    ManimColor("#E67E22"),
]


def palette_color(index: int) -> ManimColor:
    return IS_PALETTE[index % len(IS_PALETTE)]
