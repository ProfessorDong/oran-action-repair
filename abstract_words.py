#!/usr/bin/env python3
"""Count the rendered abstract's words.

Counts from the PDF, not the source, because the numeric macros expand there.
Joins wrapped lines with a SPACE -- `tr -d '\n'` glues the last word of one line
to the first of the next and silently undercounts -- and re-attaches a stranded
"%" or unit to the number it belongs to.
"""
import re, subprocess, sys

t = subprocess.run(["pdftotext", "-f", "1", "-l", "1", "main.pdf", "-"],
                   capture_output=True, text=True).stdout
a = t[t.index("Abstract"):t.index("Index Terms")]
a = a.replace("Abstract—", "").replace("Abstract-", "")
a = " ".join(a.split())
a = re.sub(r"(\d)\s+%", r"\1%", a)          # "39.2 %" is one word
n = len(a.split())
print(f"abstract: {n} words", "OK" if 75 <= n <= 200 else "OUT OF RANGE (75-200)")
sys.exit(0 if 75 <= n <= 200 else 1)
