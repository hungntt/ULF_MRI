#!/usr/bin/env bash
# Precompile the TikZ figures of the paper into figures/pdf/*.pdf.
# The paper includes these PDFs, so LaTeX does not redraw TikZ on every pass
# Re-run this after editing any *_pub_tikz.tex or any tile in figures/tiles_pub/.
set -euo pipefail
cd "$(dirname "$0")/.."            # paper/
mkdir -p figures/pdf
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for f in fig1_overview fig2_where fig3_polarity fig4_select fig5_qual; do
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$work" \
    "\def\figfile{figures/${f}_pub_tikz}\input{figures/figbuild.tex}" >/dev/null
  mv "$work/figbuild.pdf" "figures/pdf/${f}_pub.pdf"
  echo "built figures/pdf/${f}_pub.pdf"
done
for p in figures/pdf/*.pdf; do pdffonts "$p" | grep -q "Type 3" && echo "Type 3 font in $p" || true; done
