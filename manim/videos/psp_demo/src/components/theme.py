"""Light-background theme — makes the PSP demo look like a LaTeX paper figure.

Importing this module applies the background color globally via manim's
`config` singleton. Foreground colors here replace the previous uses of
``WHITE`` / ``GRAY_A`` / ``GRAY_B`` in scenes and components so text stays
legible on the light surface.

Also sets the Tex preamble (Computer Modern) so ``Tex`` and ``MathTex``
render with the standard paper font.
"""

from manim import ManimColor, TexTemplate, config

# Warm off-white — kinder on projectors than pure white.
BG_COLOR = ManimColor("#F8F7F2")

# Primary ink.
FG_PRIMARY = ManimColor("#1A1A1A")   # titles, body text
FG_SECONDARY = ManimColor("#555555") # captions, secondary labels
FG_MUTED = ManimColor("#8A8A8A")     # de-emphasized (e.g., "pool:" prefix)

# Edges on the light bg — dark enough to read against paper.
EDGE_COLOR = ManimColor("#707070")
INACTIVE_VERTEX = ManimColor("#C0C0C0")

# --- apply globally ---
config.background_color = BG_COLOR

# Tex template: use the default (Computer Modern) which ships with any
# pdflatex install. Adding amsmath/amssymb so math symbols Just Work.
_tpl = TexTemplate()
_tpl.add_to_preamble(r"\usepackage{amsmath,amssymb}")
config.tex_template = _tpl
