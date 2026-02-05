# Quantum Column Generation Slides

Animated presentation explaining quantum column generation for graph coloring using QCi Dirac-3.

## Setup

```bash
cd manim/
uv venv --python 3.12
uv sync

# macOS system deps (if needed)
brew install ffmpeg cairo pango
```

## Render All Slides

```bash
./render_all.sh                    # Low quality (fast)
./render_all.sh -q medium          # Medium quality
./render_all.sh -q high            # High quality (slow)
```

## Present

```bash
uv run manim-slides present TitleSlide GraphColoringSlide IndependentSetSlide \
  CGOverviewSlide RMPSlide PSPSlide MILPLimitationSlide QuantumIntroSlide \
  DiracExtractionSlide BenchmarkSlide SummarySlide
```

**Controls during presentation:**
- Space/Right Arrow: Next slide
- Left Arrow: Previous slide
- R: Replay current slide
- Q: Quit

## Export to HTML

```bash
uv run manim-slides convert TitleSlide GraphColoringSlide IndependentSetSlide \
  CGOverviewSlide RMPSlide PSPSlide MILPLimitationSlide QuantumIntroSlide \
  DiracExtractionSlide BenchmarkSlide SummarySlide --to html -o presentation.html
```

## Render Individual Slides

```bash
uv run manim -ql src/quantum_colgen_slides/scenes/s01_title.py TitleSlide
uv run manim -ql src/quantum_colgen_slides/scenes/s10_benchmark.py BenchmarkSlide
```

## Slide Content

| # | Scene | Description |
|---|-------|-------------|
| 1 | TitleSlide | Title and introduction |
| 2 | GraphColoringSlide | Graph coloring problem visualization |
| 3 | IndependentSetSlide | Column = Independent Set concept |
| 4 | CGOverviewSlide | CG algorithm flowchart (RMP <-> PSP) |
| 5 | RMPSlide | Restricted Master Problem LP formulation |
| 6 | PSPSlide | Pricing Subproblem (MWIS) formulation |
| 7 | MILPLimitationSlide | Why classical MILP fails (1 col/call) |
| 8 | QuantumIntroSlide | Motzkin-Straus QP and Dirac-3 |
| 9 | DiracExtractionSlide | Multi-sample extraction pipeline |
| 10 | BenchmarkSlide | Hero slide: benchmark results |
| 11 | SummarySlide | Key takeaways and scorecard |

## Key Results Shown

**ER(40,0.3):** Dirac achieves χ=6 vs Greedy χ=8 (2 colors better!)

| Method | χ | Cols/Call | Iterations |
|--------|---|-----------|------------|
| Greedy | 8 | - | - |
| MILP | 8 | 1.0 | 130 |
| LP | 8 | 10.2 | 19 |
| **Dirac** | **6** | **20.8** | **13** |

## Project Structure

```
manim/
├── pyproject.toml
├── render_all.sh
├── README.md
├── src/quantum_colgen_slides/
│   ├── scenes/           # 11 slide scenes
│   ├── components/       # Shared colors and utilities
│   └── data/             # Benchmark data
├── slides/               # Generated slide configs
└── media/                # Rendered video files
```
