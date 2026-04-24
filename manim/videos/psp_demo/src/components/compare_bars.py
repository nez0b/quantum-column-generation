"""Small grouped-bar helper for LP vs Dirac comparison scenes."""

from manim import Rectangle, Tex, Text, VGroup, UP, DOWN, RIGHT, BOLD

from .qci_colors import LP_COLOR, DIRAC_COLOR
from .theme import FG_PRIMARY, FG_SECONDARY


def bar_pair(
    label: str,
    lp_val: float,
    dirac_val: float,
    max_val: float,
    bar_width: float = 0.6,
    max_height: float = 2.2,
) -> VGroup:
    """Two side-by-side bars (LP then Dirac) with numeric labels and a caption."""
    lp_h = max(0.04, (lp_val / max_val) * max_height)
    dirac_h = max(0.04, (dirac_val / max_val) * max_height)
    lp_bar = Rectangle(
        width=bar_width, height=lp_h, fill_color=LP_COLOR, fill_opacity=0.85,
        stroke_color=LP_COLOR, stroke_width=2,
    )
    dirac_bar = Rectangle(
        width=bar_width, height=dirac_h, fill_color=DIRAC_COLOR, fill_opacity=0.85,
        stroke_color=DIRAC_COLOR, stroke_width=2,
    )
    dirac_bar.next_to(lp_bar, RIGHT, buff=0.18, aligned_edge=DOWN)
    pair = VGroup(lp_bar, dirac_bar)

    lp_num = Tex(f"{lp_val:g}", font_size=30, color=FG_PRIMARY).next_to(lp_bar, UP, buff=0.08)
    dirac_num = Tex(f"{dirac_val:g}", font_size=30, color=FG_PRIMARY).next_to(dirac_bar, UP, buff=0.08)
    cap = Tex(label, font_size=28, color=FG_SECONDARY).next_to(pair, DOWN, buff=0.25)
    return VGroup(pair, lp_num, dirac_num, cap)


def legend() -> VGroup:
    legend_lp = VGroup(
        Rectangle(width=0.25, height=0.25, fill_color=LP_COLOR, fill_opacity=0.85, stroke_width=0),
        Tex("LP", font_size=30, color=LP_COLOR),
    ).arrange(RIGHT, buff=0.15)
    legend_dirac = VGroup(
        Rectangle(width=0.25, height=0.25, fill_color=DIRAC_COLOR, fill_opacity=0.85, stroke_width=0),
        Tex("Dirac", font_size=30, color=DIRAC_COLOR),
    ).arrange(RIGHT, buff=0.15)
    return VGroup(legend_lp, legend_dirac).arrange(RIGHT, buff=0.6)
