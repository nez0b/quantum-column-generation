#!/usr/bin/env bash
# build.sh — compile deep_dive.pdf. Run twice for frame numbering.
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode deep_dive.tex >/dev/null
pdflatex -interaction=nonstopmode deep_dive.tex >/dev/null
echo "deep_dive.pdf built ($(du -h deep_dive.pdf | cut -f1))"
