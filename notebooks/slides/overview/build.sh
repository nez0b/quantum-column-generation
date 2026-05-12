#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode overview.tex >/dev/null
pdflatex -interaction=nonstopmode overview.tex >/dev/null
echo "overview.pdf built ($(du -h overview.pdf | cut -f1))"
