#!/bin/bash
# Render the CG vs QCG PSP demo (p01-p12) and concat into a single mp4.
#
# Usage:  ./render_psp_demo.sh [low|medium|high]   (default: medium)
#
set -e

QUALITY="${1:-medium}"
case "$QUALITY" in
    low)    QFLAG="-ql"; QDIR="480p15" ;;
    medium) QFLAG="-qm"; QDIR="720p30" ;;
    high)   QFLAG="-qh"; QDIR="1080p60" ;;
    *)      echo "Unknown quality: $QUALITY"; exit 1 ;;
esac

SCENES=(
    "p01_psp_title.py:PSPTitleScene"
    "p02_graph_intro.py:PSPGraphIntroScene"
    "p03_psp1_duals.py:PSP1DualsScene"
    "p04_psp1_lp_flash.py:PSP1LPFlashScene"
    "p05_psp1_dirac_flash.py:PSP1DiracFlashScene"
    "p06_psp1_compare.py:PSP1CompareScene"
    "p07_psp2_duals.py:PSP2DualsScene"
    "p08_psp2_lp_flash.py:PSP2LPFlashScene"
    "p09_psp2_dirac_flash.py:PSP2DiracFlashScene"
    "p10_psp2_compare.py:PSP2CompareScene"
    "p11_aggregate_summary.py:AggregateSummaryScene"
    "p12_closing.py:ClosingScene"
)

cd "$(dirname "$0")"

echo "[render] quality=$QUALITY ($QDIR)"
for s in "${SCENES[@]}"; do
    file="${s%%:*}"
    scene="${s##*:}"
    echo "  · $scene from $file"
    uv run manim $QFLAG "src/quantum_colgen_slides/scenes/$file" "$scene"
done

# Build concat list.
LIST="$(mktemp)"
trap "rm -f $LIST" EXIT
for s in "${SCENES[@]}"; do
    file="${s%%:*}"
    scene="${s##*:}"
    base="${file%.py}"
    # Manim path:  media/videos/<module-stem>/<QDIR>/<SceneClass>.mp4
    mp4="media/videos/${base}/${QDIR}/${scene}.mp4"
    if [[ ! -f "$mp4" ]]; then
        echo "[err] missing mp4: $mp4" >&2
        exit 1
    fi
    echo "file '$PWD/$mp4'" >> "$LIST"
done

OUT="qcg_vs_cg_demo.mp4"
echo "[concat] → $OUT"
ffmpeg -y -loglevel warning -f concat -safe 0 -i "$LIST" -c copy "$OUT"
echo "[done] $OUT ($(wc -c < "$OUT" | awk '{printf "%.1f MB\n", $1/1048576}'))"
