#!/bin/bash
# Render all slides for the Quantum Column Generation presentation
# Usage: ./render_all.sh [-q|--quality low|medium|high]

set -e

QUALITY="${1:-low}"
QUALITY_FLAG="-ql"  # default to low quality (480p15)

case "$QUALITY" in
    "-q"|"--quality")
        shift
        case "$1" in
            "low")     QUALITY_FLAG="-ql" ;;
            "medium")  QUALITY_FLAG="-qm" ;;
            "high")    QUALITY_FLAG="-qh" ;;
            *)         echo "Unknown quality: $1. Use low, medium, or high." && exit 1 ;;
        esac
        ;;
esac

echo "Rendering slides with quality flag: $QUALITY_FLAG"

# Render each slide
SLIDES=(
    "s01_title.py:TitleSlide"
    "s02_graph_coloring.py:GraphColoringSlide"
    "s03_independent_set.py:IndependentSetSlide"
    "s04_cg_overview.py:CGOverviewSlide"
    "s05_rmp.py:RMPSlide"
    "s06_psp.py:PSPSlide"
    "s07_milp_limitation.py:MILPLimitationSlide"
    "s08_quantum_intro.py:QuantumIntroSlide"
    "s09_dirac_extraction.py:DiracExtractionSlide"
    "s10_benchmark.py:BenchmarkSlide"
    "s10a_er40_graph.py:ER40GraphSlide"
    "s10b_er40_colorings.py:ER40ColoringsSlide"
    "s10c_er50_graph.py:ER50GraphSlide"
    "s10d_er50_colorings.py:ER50ColoringsSlide"
    "s11_summary.py:SummarySlide"
)

for slide in "${SLIDES[@]}"; do
    file="${slide%%:*}"
    scene="${slide##*:}"
    echo "Rendering $scene from $file..."
    uv run manim $QUALITY_FLAG "src/quantum_colgen_slides/scenes/$file" "$scene"
done

echo ""
echo "All slides rendered successfully!"
echo ""
echo "To present the slides:"
echo "  uv run manim-slides present TitleSlide GraphColoringSlide IndependentSetSlide \\"
echo "    CGOverviewSlide RMPSlide PSPSlide MILPLimitationSlide QuantumIntroSlide \\"
echo "    DiracExtractionSlide BenchmarkSlide ER40GraphSlide ER40ColoringsSlide \\"
echo "    ER50GraphSlide ER50ColoringsSlide SummarySlide"
echo ""
echo "To export to HTML:"
echo "  uv run manim-slides convert TitleSlide GraphColoringSlide IndependentSetSlide \\"
echo "    CGOverviewSlide RMPSlide PSPSlide MILPLimitationSlide QuantumIntroSlide \\"
echo "    DiracExtractionSlide BenchmarkSlide ER40GraphSlide ER40ColoringsSlide \\"
echo "    ER50GraphSlide ER50ColoringsSlide SummarySlide --to html -o presentation.html"
