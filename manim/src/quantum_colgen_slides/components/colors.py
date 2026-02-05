"""Color palette for the presentation."""

from manim import *

# Graph coloring palette (colorblind-friendly)
GRAPH_COLORS = [
    "#E63946",  # Red
    "#457B9D",  # Blue
    "#2A9D8F",  # Teal/Green
    "#E9C46A",  # Yellow
    "#F4A261",  # Orange
    "#9B5DE5",  # Purple
    "#00F5D4",  # Cyan
    "#FEE440",  # Bright Yellow
]

# Convert to manim color format
GRAPH_RED = ManimColor(GRAPH_COLORS[0])
GRAPH_BLUE = ManimColor(GRAPH_COLORS[1])
GRAPH_GREEN = ManimColor(GRAPH_COLORS[2])
GRAPH_YELLOW = ManimColor(GRAPH_COLORS[3])
GRAPH_ORANGE = ManimColor(GRAPH_COLORS[4])
GRAPH_PURPLE = ManimColor(GRAPH_COLORS[5])

# Slide theme colors
TITLE_COLOR = WHITE
SUBTITLE_COLOR = GRAY_A
HIGHLIGHT_COLOR = YELLOW
BOX_COLOR_RMP = BLUE_C
BOX_COLOR_PSP = GREEN_C
BOX_COLOR_DIRAC = PURPLE_A

# Benchmark table colors
WIN_COLOR = GREEN
TIE_COLOR = YELLOW
LOSS_COLOR = RED
NEUTRAL_COLOR = WHITE


def get_graph_color(index: int) -> ManimColor:
    """Get a graph coloring color by index (wraps around)."""
    hex_color = GRAPH_COLORS[index % len(GRAPH_COLORS)]
    return ManimColor(hex_color)
