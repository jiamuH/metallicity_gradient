#!/usr/bin/env python3
"""Quick check: how much does the alpha_ox uncertainty move the inferred N?

At fixed observed ratio the required nitrogen scale shifts by the change in
the model ratio (ratios are near-linear in N at trace level), so
|d log N| ~= |d log(ratio)/d(aox)| * delta_aox.  Print the local slope around
the low-ion-consistent aox (-1.65) at solar Z for N V, N IV], N III] / C IV.
"""
import numpy as np
from plot_aox_line_ratios import load_grid

records = load_grid()
sel = sorted([r for r in records if abs(r['logZ']) < 1e-3],
             key=lambda r: r['aox'])
aox = np.array([r['aox'] for r in sel])
print('aox grid at solar Z:', aox)
for key in ('NV', 'NIV', 'NIII'):
    ratio = np.array([r['lines'][key] / r['lines']['CIV'] for r in sel])
    lr = np.log10(ratio)
    i = np.searchsorted(aox, -1.65)
    slope = (lr[i] - lr[i - 1]) / (aox[i] - aox[i - 1])
    print(f'{key}/CIV: dlog(ratio)/d(aox) ~ {slope:+.2f}   '
          f'per 0.1 in aox: {abs(slope)*0.1:.3f} dex   '
          f'across strip 0.04: {abs(slope)*0.04:.3f} dex')
