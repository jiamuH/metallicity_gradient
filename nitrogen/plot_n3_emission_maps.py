"""N III] 1750 emission over the (log phi, log n_H) cloud grid at Z_alpha = 1
and 3 Z_sun (fixed-nitrogen N23 grids, nitrogen at solar N/alpha).  Colour is
log(F / phi), the N III] flux per incident ionizing photon, which is what the
LOC radius calculation weights.  The dashed box marks the cloud ensemble used
for the abundances and radii (log phi 17-21, log n_H >= 9).

Run (from the project root): python3 nitrogen/plot_n3_emission_maps.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import loc_emission_weighted_r as L
from loc_nitrogen_abundance import PHI_COMMON

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy', 'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

OUT = 'plots/nitrogen/n3_emission_maps_Z1_Z3.png'
PHI = np.arange(17.0, 23.5, 0.5)
LOGN = np.arange(9.0, 12.5, 0.5)
GRIDS = [(r'$Z_\alpha = Z_\odot$', 'strong_LOC_m1_N23_LineList_BLR_Fe2_flux.txt'),
         (r'$Z_\alpha = 3\,Z_\odot$', 'strong_LOC_m3_N23_LineList_BLR_Fe2_flux.txt')]


def n3_map(fname):
    """log10(F_N3 / phi) on (PHI, LOGN)."""
    em = L._load_line_intensity([(fname, PHI, LOGN)])['N3']
    ip = [int(np.argmin(abs(L.PHI_AX_FULL - p))) for p in PHI]
    inn = [int(np.argmin(abs(L.LOGN_AX_FULL - n))) for n in LOGN]
    f = em[np.ix_(ip, inn)]
    with np.errstate(divide='ignore'):
        return np.log10(np.maximum(f, 1e-30) / 10 ** PHI[:, None])


def main():
    maps = [n3_map(f) for _, f in GRIDS]
    vmax = max(np.nanmax(m) for m in maps)
    vmin = vmax - 4.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True,
                             gridspec_kw=dict(wspace=0.06))
    for ax, (label, _), m in zip(axes, GRIDS, maps):
        im = ax.pcolormesh(PHI, LOGN, m.T, vmin=vmin, vmax=vmax, cmap='magma', shading='nearest')
        ax.contour(PHI, LOGN, m.T, levels=[vmax - 1, vmax - 0.5], colors=['0.8', 'white'],
                   linewidths=[1.5, 2.2])
        # emission-weighted log phi of N III] inside the ensemble (flat weight in log n)
        sel = (PHI >= PHI_COMMON[0]) & (PHI <= PHI_COMMON[1])
        w = (10 ** m[sel]).mean(axis=1)
        phi_ew = np.sum(PHI[sel] * w) / np.sum(w)
        ax.axvline(phi_ew, color='cyan', lw=2.2, ls='-')
        ax.plot([PHI_COMMON[0], PHI_COMMON[1], PHI_COMMON[1], PHI_COMMON[0], PHI_COMMON[0]],
                [9, 9, 12, 12, 9], '--', color='white', lw=1.5)
        ax.text(0.04, 0.95, label, transform=ax.transAxes, color='white', va='top')
        ax.text(0.04, 0.85, rf'$\langle\log\phi\rangle_{{\rm N\,III]}} = {phi_ew:.2f}$',
                transform=ax.transAxes, color='cyan', va='top', fontsize=16)
        ax.set_xlabel(r'$\log \phi~[\rm cm^{-2}~s^{-1}]$')
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', direction='in', top=True, right=True,
                       length=8, width=1.5, color='white')
        ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True,
                       length=4, width=1, color='white')
        print(f'{label}: emission-weighted log phi of N III] in the ensemble = {phi_ew:.2f}')
    axes[0].set_ylabel(r'$\log n_{\rm H}~[\rm cm^{-3}]$')
    cb = fig.colorbar(im, ax=axes, pad=0.015)
    cb.set_label(r'$\log~(F_{\rm N\,III]}/\phi)$')
    cb.ax.tick_params(direction='in')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
