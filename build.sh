#!/usr/bin/env bash
# Regenerate figures, tables and numeric macros from the result CSVs, then
# build the manuscript.  Does NOT re-run the simulations; see README.md for
# the commands that produce sim/journal/results/.
set -euo pipefail
cd "$(dirname "$0")"

echo "== figures =="
(cd sim/journal && python3 figures.py)

echo
echo "== numbers, tables, and consistency checks =="
(cd sim/journal && python3 make_numbers.py)

# The manuscript is not published in this repository, so the LaTeX step runs
# only when main.tex is present locally.
if [ -f main.tex ]; then
  echo
  echo "== latex =="
  latexmk -quiet -pdf main.tex

  echo
  echo "Built main.pdf ($(pdfinfo main.pdf | awk '/^Pages/{print $2}') pages)."

  python3 abstract_words.py || echo "  [FAIL] abstract is outside the 75-200 word range"
else
  echo
  echo "== latex: skipped (main.tex not present) =="
  echo "Figures, numbers.tex, the tables and the consistency checks are current."
fi
