"""
NV/CIV (and (SiIV+OIV)/CIV) as a function of ionizing flux phi, from the Cloudy
LOC grid with the density axis collapsed (marginalized) -- the same n-average
used in the breathing model.  phi is the control variable here (n is integrated
out, so U is not a single value).

Run:
    ~/miniconda3/envs/pypeit/bin/python alpha/plot_nv_civ_vs_phi.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from loc_gradient_fit import load_cloudy_grid, LOGPHI_GRID, Z_GRID

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy',
                     'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

PLOT_DIR = "plots/alpha/loc_grid"
os.makedirs(PLOT_DIR, exist_ok=True)

# load grid {line: (n, phi, Z)} and collapse density by a simple mean over n
arr = load_cloudy_grid()
coll = {k: np.mean(v, axis=0) for k, v in arr.items()}   # (phi, Z)
NV_CIV = coll['N5'] / coll['C4']                          # (phi, Z)
SIO_CIV = (coll['Si4'] + coll['O4']) / coll['C4']

Z_TARGETS = [1.0, 3.0, 6.0, 10.0]
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']

fig, ax = plt.subplots(figsize=(8, 7))
for zt, c in zip(Z_TARGETS, colors):
    j = int(np.argmin(np.abs(Z_GRID - zt)))
    ax.plot(LOGPHI_GRID, np.log10(NV_CIV[:, j]), 'o-', color=c, lw=2.5, ms=7,
            label=r'$Z=%.0f\,Z_\odot$' % Z_GRID[j])

ax.set_xlabel(r'$\log~\phi~[{\rm photons~cm^{-2}~s^{-1}}]$')
ax.set_ylabel(r'$\log~(\mathrm{NV/CIV})$')
ax.minorticks_on()
ax.tick_params(which='major', direction='in', length=8, width=1.5,
               top=True, right=True)
ax.tick_params(which='minor', direction='in', length=4, width=1.0,
               top=True, right=True)
ax.legend(fontsize=14, loc='best')
fig.tight_layout()
out = os.path.join(PLOT_DIR, 'nv_civ_vs_phi.png')
fig.savefig(out, dpi=200)
plt.close(fig)
print(f"-> wrote {out}")

# print the numbers at Z=3
j3 = int(np.argmin(np.abs(Z_GRID - 3.0)))
print(f"\nZ=3 Zsun, n-collapsed:")
print(f"  {'log phi':>8} {'log(NV/CIV)':>12} {'log((SiIV+OIV)/CIV)':>20}")
for i, lp in enumerate(LOGPHI_GRID):
    print(f"  {lp:8.1f} {np.log10(NV_CIV[i, j3]):12.3f} "
          f"{np.log10(SIO_CIV[i, j3]):20.3f}")
