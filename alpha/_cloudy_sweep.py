#!/usr/bin/env python3
"""Standardize `Cloudy` -> `\\textsc{Cloudy}` in metal_gradient.tex.

- replaces \\texttt{CLOUDY}  -> \\textsc{Cloudy}
- replaces plain `Cloudy` (word-bounded) -> \\textsc{Cloudy},
  skipping anything already preceded by `\\textsc{` or part of an
  identifier like `Cloudy_nitrogen_...` (filenames).
- idempotent.
"""
import re
from pathlib import Path

P = Path('Diffusion/metal_gradient.tex')
text = P.read_text()
orig = text

text = text.replace(r'\texttt{CLOUDY}', r'\textsc{Cloudy}')
text = re.sub(r'(?<!\\textsc\{)\bCloudy\b', r'\\textsc{Cloudy}', text)

n_lines = sum(1 for a, b in zip(orig.splitlines(), text.splitlines()) if a != b)
n_sub = orig.count('Cloudy') - text.replace(r'\textsc{Cloudy}', 'X').count('Cloudy')
print(f'lines changed:                       {n_lines}')
P.write_text(text)
