"""Debug: the highest-EW cells and local EW maxima of N III] 1750 on the 3 Z_sun grid
(the top-left panel of Figure 2).

Run: python3 -m nitrogen._check_n3_ew_maxima
"""
import os

import numpy as np

from nitrogen.plot_loc_m3_N23_contours import TAG_INFO, GRID_DIR, LINE_INFO, _read_grid

fname, phi_ax, n_ax = TAG_INFO['m3_N23']['chunks'][0]
df = _read_grid(os.path.join(GRID_DIR, fname))
lc = LINE_INFO['N3']
ew = (df[lc['col']].astype(float) / df[lc['cont']].astype(float).clip(lower=1e-30)).values
ew = ew.reshape(len(phi_ax), len(n_ax))

print('Top 8 cells (log phi, log n, log EW):')
for k in np.argsort(ew.ravel())[::-1][:8]:
    i, j = np.unravel_index(k, ew.shape)
    print(f'  {phi_ax[i]:5.1f} {n_ax[j]:5.1f}  {np.log10(ew[i, j]):6.3f}')

print('Local maxima (greater than all 8 neighbours):')
for i in range(len(phi_ax)):
    for j in range(len(n_ax)):
        nb = ew[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2]
        if ew[i, j] > 0 and ew[i, j] == nb.max():
            print(f'  {phi_ax[i]:5.1f} {n_ax[j]:5.1f}  {np.log10(ew[i, j]):6.3f}')
