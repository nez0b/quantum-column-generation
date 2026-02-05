# Manim Slides Usage Guide

Complete guide for editing, rendering, and exporting the quantum column generation presentation.

## Prerequisites

```bash
cd manim/
uv venv --python 3.12
uv sync

# macOS system dependencies
brew install ffmpeg cairo pango
```

## Project Structure

```
manim/
├── src/quantum_colgen_slides/
│   ├── scenes/           # Slide scene files (s01-s11)
│   ├── components/       # Shared colors, graph utilities
│   └── data/             # Benchmark data module
├── render_all.sh         # Batch render script
├── slides/               # Generated .json slide configs
└── media/                # Rendered video files
```

## Editing Slides

### Scene Files

Each slide is a Python class extending `Slide` from `manim_slides`:

```
scenes/
├── s01_title.py          # TitleSlide
├── s02_graph_coloring.py # GraphColoringSlide
├── s03_independent_set.py# IndependentSetSlide
├── s04_cg_overview.py    # CGOverviewSlide
├── s05_rmp.py            # RMPSlide
├── s06_psp.py            # PSPSlide
├── s07_milp_limitation.py# MILPLimitationSlide
├── s08_quantum_intro.py  # QuantumIntroSlide
├── s09_dirac_extraction.py# DiracExtractionSlide
├── s10_benchmark.py      # BenchmarkSlide (main results)
├── s10a_er40_graph.py    # ER40GraphSlide
├── s10b_er40_colorings.py# ER40ColoringsSlide
├── s10c_er50_graph.py    # ER50GraphSlide
├── s10d_er50_colorings.py# ER50ColoringsSlide
└── s11_summary.py        # SummarySlide
```

### Modifying Content

1. **Text/equations**: Edit the relevant `Text()` or `MathTex()` calls
2. **Colors**: Modify `components/colors.py`
3. **Graph layouts**: Adjust `components/graph_utils.py`
4. **Benchmark data**: Update `data/benchmark_data.py`

### Adding Animation Breaks

Use `self.next_slide()` to create pause points during presentation:

```python
def construct(self):
    title = Text("My Slide")
    self.play(Write(title))
    self.next_slide()  # Pause here - press Space to continue

    content = Text("More content")
    self.play(FadeIn(content))
    self.next_slide()  # Another pause
```

## Rendering

### Single Slide

```bash
# Low quality (fast iteration)
uv run manim -ql src/quantum_colgen_slides/scenes/s01_title.py TitleSlide

# Medium quality
uv run manim -qm src/quantum_colgen_slides/scenes/s01_title.py TitleSlide

# High quality (production)
uv run manim -qh src/quantum_colgen_slides/scenes/s01_title.py TitleSlide

# 4K quality
uv run manim -qk src/quantum_colgen_slides/scenes/s01_title.py TitleSlide
```

### All Slides

```bash
./render_all.sh           # Low quality (default)
./render_all.sh -q medium # Medium quality
./render_all.sh -q high   # High quality
```

### Quality Settings

| Flag | Resolution | FPS | Use Case |
|------|------------|-----|----------|
| `-ql` | 854x480 | 15 | Quick preview |
| `-qm` | 1280x720 | 30 | Draft review |
| `-qh` | 1920x1080 | 60 | Final presentation |
| `-qk` | 3840x2160 | 60 | 4K export |

## Presenting

### Interactive Presentation

```bash
uv run manim-slides present TitleSlide GraphColoringSlide IndependentSetSlide \
  CGOverviewSlide RMPSlide PSPSlide MILPLimitationSlide QuantumIntroSlide \
  DiracExtractionSlide BenchmarkSlide SummarySlide
```

### Presentation Controls

| Key | Action |
|-----|--------|
| Space / Right Arrow | Next slide/animation |
| Left Arrow | Previous slide |
| R | Replay current slide |
| F | Toggle fullscreen |
| Q | Quit presentation |

### Present Specific Slides

```bash
# Just benchmark slides
uv run manim-slides present ER40GraphSlide ER40ColoringsSlide ER50GraphSlide ER50ColoringsSlide
```

## Export Formats

### HTML (Interactive)

Creates a self-contained HTML file with embedded videos:

```bash
uv run manim-slides convert TitleSlide GraphColoringSlide IndependentSetSlide \
  CGOverviewSlide RMPSlide PSPSlide MILPLimitationSlide QuantumIntroSlide \
  DiracExtractionSlide BenchmarkSlide SummarySlide --to html -o presentation.html
```

### PowerPoint

```bash
uv run manim-slides convert TitleSlide GraphColoringSlide IndependentSetSlide \
  CGOverviewSlide RMPSlide PSPSlide MILPLimitationSlide QuantumIntroSlide \
  DiracExtractionSlide BenchmarkSlide SummarySlide --to pptx -o presentation.pptx
```

### PDF (Static Frames)

```bash
uv run manim-slides convert TitleSlide GraphColoringSlide IndependentSetSlide \
  CGOverviewSlide RMPSlide PSPSlide MILPLimitationSlide QuantumIntroSlide \
  DiracExtractionSlide BenchmarkSlide SummarySlide --to pdf -o presentation.pdf
```

### Video (MP4)

Export as a continuous video file:

```bash
# Render high quality first
./render_all.sh -q high

# Concatenate videos (requires ffmpeg)
ffmpeg -f concat -safe 0 -i <(for f in media/videos/*/1080p60/*.mp4; do echo "file '$PWD/$f'"; done) \
  -c copy presentation.mp4
```

## Updating Benchmark Data

### From JSON Results

The benchmark data in `data/benchmark_data.py` can be updated from the `benchmarks/` folder:

```python
# data/benchmark_data.py
@dataclass
class BenchmarkResult:
    method: str        # "greedy", "lp", "dirac"
    graph_name: str    # "ER(40,0.3)"
    n_nodes: int
    n_edges: int
    chi: int           # chromatic number found
    iterations: Optional[int] = None
    cols_per_call: Optional[float] = None
    wall_seconds: Optional[float] = None
```

### Adding New Graph Results

1. Run benchmarks: `uv run python scripts/run_benchmark_full.py --graphs 75 100`
2. Update `data/benchmark_data.py` with new results
3. Create new scene files (e.g., `s10e_er75_graph.py`)
4. Add to `render_all.sh` and presentation command

## Troubleshooting

### Missing LaTeX

Install LaTeX for math rendering:
```bash
brew install --cask mactex-no-gui  # macOS
```

### Slow Rendering

- Use `-ql` during development
- Disable anti-aliasing: `config.disable_caching = True`
- Reduce frame rate: `config.frame_rate = 15`

### Font Issues

Install required fonts:
```bash
# macOS
brew tap homebrew/cask-fonts
brew install --cask font-source-sans-pro
```

### Video Not Playing

Ensure ffmpeg is installed and codecs are available:
```bash
ffmpeg -encoders | grep 264
```
