#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode tutorial.tex >/dev/null
pdflatex -interaction=nonstopmode tutorial.tex >/dev/null
echo "tutorial.pdf built ($(du -h tutorial.pdf | cut -f1))"
