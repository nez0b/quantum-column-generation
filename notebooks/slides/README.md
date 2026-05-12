# `slides/` — companion presentations

Three beamer-qci-styled PDF presentations, each matched to a notebook
audience.

| Deck | File | Audience | Pages |
|---|---|---|---|
| **Deep dive** | `deep_dive/deep_dive.pdf` | Technical (engineering / research). Mirrors notebooks `01_motzkin_straus`, `02a/b_column_generation_*`, `03a/b_application_demo_*`. | 24 |
| **Tutorial** | `tutorial/tutorial.pdf` | Technical customer. Mirrors notebook `04_tutorial_end_to_end` — same narrative arc end-to-end, condensed. | 14 |
| **Overview** | `overview/overview.pdf` | High-level customer / executive pitch. No math, no algorithms — just the value proposition. | 9 |

## Rebuilding the decks

Each deck has its own `build.sh`:

```bash
cd notebooks/slides/deep_dive && ./build.sh
cd notebooks/slides/tutorial  && ./build.sh
cd notebooks/slides/overview  && ./build.sh
```

Or rebuild all three from the top level:

```bash
make -C notebooks/slides
```

Requires `pdflatex` (TeX Live 2025 or later). Each `build.sh` runs
pdflatex twice for frame numbers.

## Style

All decks use the canonical QCi beamer template from
`~/Code/slide-templates/beamer-qci/`:

- Madrid theme with custom QCi navigation bar (`palette primary` =
  `QCIcol1`, the brand dark purple-blue).
- Raleway font.
- Title page with QCi logo at top and company info footer.
- White QCi logo in the section-navigation header on every frame.
- Block colors: `qciblue / qciteal / qciorange / qcigreen` for content
  accent; `QCIcol1-5` for background/structure.

The decks share logo assets (`QCIlogo2.pdf`, `QCILogoWhite.png`)
duplicated into each deck folder so each is self-buildable. The
`shared/` folder is the original copy.

## Figures

- `deep_dive/figures/*.pdf` — 12 pre-built figures vendored from the
  original `RF-branching/slides/qcg_vs_cg_demo/figures/` (graph view,
  PSP duals, LP/Dirac columns, statistics, summary bars, B&P node
  counts, full benchmark table).
- `tutorial/figures/` and `overview/figures/` — empty; both decks
  generate their diagrams inline with TikZ.
