"""Flash a sequence of independent sets on a shared graph — the animation centerpiece.

The visual idiom: for each column, draw a semi-transparent convex-hull polygon
behind the IS vertices and fill those vertices with the column's palette color.
Hold briefly, then fade out. A persistent counter updates in the corner and a
pool strip of compact ``{v,...}`` indicators grows along the bottom.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from manim import (
    DOWN,
    FadeIn,
    FadeOut,
    Line,
    ManimColor,
    MathTex,
    Polygon,
    ReplacementTransform,
    Scene,
    Tex,
    Text,
    VGroup,
    UP,
    LEFT,
    RIGHT,
    Dot,
    SurroundingRectangle,
)
from manim.mobject.graph import Graph

from .qci_colors import IS_PALETTE, palette_color
from .theme import FG_PRIMARY, FG_SECONDARY


# ── hull geometry ──────────────────────────────────────────────────

def _sorted_hull(points: np.ndarray) -> np.ndarray:
    """Return points sorted by angle around centroid (stable for IS subsets
    on a 2D layout — they are typically in convex position)."""
    c = points.mean(axis=0)
    ang = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
    order = np.argsort(ang)
    return points[order]


def _is_overlay(
    graph: Graph,
    column: Iterable[int],
    color: ManimColor,
    pad: float = 0.28,
) -> VGroup:
    """Draw a convex hull polygon + vertex highlights for the given column."""
    column = list(column)
    points = np.array([graph.vertices[v].get_center()[:2] for v in column])

    hull_group = VGroup()
    if len(points) == 1:
        halo = Dot(
            point=[*points[0], 0],
            radius=pad,
            fill_color=color,
            fill_opacity=0.32,
            stroke_color=color,
            stroke_width=2.5,
        )
        hull_group.add(halo)
    elif len(points) == 2:
        line = Line(
            start=[*points[0], 0],
            end=[*points[1], 0],
            stroke_color=color,
            stroke_width=14,
            stroke_opacity=0.42,
        )
        hull_group.add(line)
    else:
        ordered = _sorted_hull(points)
        c = ordered.mean(axis=0)
        padded = []
        for p in ordered:
            d = p - c
            n = np.linalg.norm(d)
            if n < 1e-9:
                padded.append([*p, 0])
            else:
                padded.append([*(p + d * (pad / n)), 0])
        hull = Polygon(
            *padded,
            fill_color=color,
            fill_opacity=0.32,
            stroke_color=color,
            stroke_width=3,
        )
        hull_group.add(hull)
    # Ring stroke on each IS vertex so the group membership stays legible.
    for v in column:
        vmob = graph.vertices[v]
        ring = SurroundingRectangle(
            vmob,
            color=color,
            buff=0.03,
            corner_radius=vmob.width / 2,
            stroke_width=3,
        )
        hull_group.add(ring)
    return hull_group


# ── pool strip (growing list of compact {v,...} chips) ──────────────

def _chip(column: Sequence[int], color: ManimColor, font_size: int = 18) -> VGroup:
    """Compact ``\\{v,w,...\\}`` label inside a rounded box — echo of the beamer small-multiples."""
    inner = ",".join(str(v) for v in sorted(column))
    lbl = MathTex(r"\{" + inner + r"\}", font_size=font_size, color=FG_PRIMARY)
    box = SurroundingRectangle(
        lbl, color=color, corner_radius=0.06, buff=0.08, stroke_width=1.5,
    )
    box.set_fill(color, opacity=0.18)
    return VGroup(box, lbl)


# ── counter ──────────────────────────────────────────────────────────

def _make_counter(oracle_label: str, total: int, color: ManimColor) -> VGroup:
    head = Tex(rf"\textbf{{{oracle_label} columns}}", font_size=38, color=color)
    num = Text(f"0 / {total}", font_size=40, color=FG_PRIMARY, weight="BOLD")
    num.next_to(head, DOWN, buff=0.2)
    return VGroup(head, num)


# ── public entry point ──────────────────────────────────────────────

def flash_columns(
    scene: Scene,
    graph: Graph,
    columns: Sequence[Sequence[int]],
    oracle_label: str,
    oracle_color: ManimColor,
    counter_position,
    pool_anchor,
    dwell: float = 0.9,
    hold: float = 0.25,
    fade_time: float = 0.25,
    pool_cols_per_row: int = 8,
    pool_row_spacing: float = 0.35,
) -> Tuple[VGroup, VGroup]:
    """Flash each column in sequence on ``graph``; update a counter and pool strip.

    Choreography choice: discrete pulses (fade-in → hold → fade-out) so each
    column is individually countable. The growing pool strip below the graph
    preserves the cumulative view that pure pulses would lose.

    Returns (counter_mobject, pool_strip_mobject) so later scenes can reuse them.
    """
    total = len(columns)
    counter = _make_counter(oracle_label, total, oracle_color).move_to(counter_position)
    scene.play(FadeIn(counter, run_time=0.4))

    pool = VGroup().move_to(pool_anchor)
    scene.add(pool)

    chips: List[VGroup] = []

    for i, col in enumerate(columns):
        color = palette_color(i)
        overlay = _is_overlay(graph, col, color)
        chip = _chip(col, color)

        # Position chip: first chip at anchor; new row below first chip of prior row;
        # subsequent chips in a row to the right of the prior chip.
        row = i // pool_cols_per_row
        col_idx = i % pool_cols_per_row
        if i == 0:
            chip.move_to(pool_anchor).align_to(pool_anchor, LEFT)
        elif col_idx == 0:
            first_of_prev_row = chips[(row - 1) * pool_cols_per_row]
            chip.next_to(first_of_prev_row, DOWN, buff=pool_row_spacing, aligned_edge=LEFT)
        else:
            chip.next_to(chips[-1], RIGHT, buff=0.1)
        chips.append(chip)
        pool.add(chip)

        # Update counter label in-place via ReplacementTransform.
        new_num = Text(f"{i+1} / {total}", font_size=36, color=FG_PRIMARY, weight="BOLD")
        new_num.move_to(counter[1])

        scene.play(
            FadeIn(overlay),
            FadeIn(chip, shift=UP * 0.1),
            ReplacementTransform(counter[1], new_num),
            run_time=fade_time,
        )
        counter.submobjects[1] = new_num
        scene.wait(hold)

        # Fade the overlay out (keep the chip in the pool).
        scene.play(FadeOut(overlay), run_time=fade_time)

    return counter, pool
