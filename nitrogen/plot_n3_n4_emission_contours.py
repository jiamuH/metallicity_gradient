"""Where do N III] 1750 and N IV] 1486 come from in the (log phi, log n_H)
cloud grid at Z_alpha = 1 and 3 Z_sun?  Contours of log(F/phi) for each line,
each relative to its own peak (solid: 0.5 dex below, thin: 1 dex below), with
the emission-weighted log phi of each line inside the LOC ensemble (dashed box:
log phi 17-21, log n_H 9-12) marked by vertical lines.

Run (from the project root): python3 nitrogen/plot_n3_n4_emission_contours.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import loc_emission_weighted_r as L
from loc_nitrogen_abundance import PHI_COMMON

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy', 'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

OUT = 'plots/nitrogen/n3_n4_emission_contours_Z1_Z3.png'
PHI = np.arange(17.0, 23.5, 0.5)
LOGN = np.arange(9.0, 12.5, 0.5)
GRIDS = [(r'$Z_\alpha = Z_\odot$', 'strong_LOC_m1_N23_LineList_BLR_Fe2_flux.txt'),
         (r'$Z_\alpha = 3\,Z_\odot$', 'strong_LOC_m3_N23_LineList_BLR_Fe2_flux.txt')]
IONS = {'N4': (r'$\rm N\,IV]$', '#C2185B'), 'N3': (r'$\rm N\,III]$', '#5B2B7B')}


def maps(fname):
    """{ion: log10(F / phi)} on (PHI, LOGN)."""
    em = L._load_line_intensity([(fname, PHI, LOGN)])
    ip = [int(np.argmin(abs(L.PHI_AX_FULL - p))) for p in PHI]
    inn = [int(np.argmin(abs(L.LOGN_AX_FULL - n))) for n in LOGN]
    return {ion: np.log10(np.maximum(em[ion][np.ix_(ip, inn)], 1e-30) / 10 ** PHI[:, None])
            for ion in IONS}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, gridspec_kw=dict(wspace=0.06))
    sel = (PHI >= PHI_COMMON[0]) & (PHI <= PHI_COMMON[1])
    for ax, (label, fname) in zip(axes, GRIDS):
        m = maps(fname)
        for ion, (lab, col) in IONS.items():
            peak = np.nanmax(m[ion][sel])
            ax.contour(PHI, LOGN, m[ion].T, levels=[peak - 1.0], colors=[col], linewidths=1.5, linestyles='solid')
            ax.contourf(PHI, LOGN, m[ion].T, levels=[peak - 0.5, peak + 1], colors=[col], alpha=0.25)
            ax.contour(PHI, LOGN, m[ion].T, levels=[peak - 0.5], colors=[col], linewidths=3, linestyles='solid')
            w = (10 ** m[ion][sel]).mean(axis=1)
            phi_ew = np.sum(PHI[sel] * w) / np.sum(w)
            ax.axvline(phi_ew, color=col, lw=2.2, ls=(0, (4, 2)))
            print(f'{label}  {ion}: emission-weighted log phi = {phi_ew:.2f}')
        ax.plot([PHI_COMMON[0], PHI_COMMON[1], PHI_COMMON[1], PHI_COMMON[0], PHI_COMMON[0]],
                [9, 9, 12, 12, 9], '--', color='0.5', lw=1.3)
        ax.text(0.96, 0.95, label, transform=ax.transAxes, ha='right', va='top')
        ax.set_xlabel(r'$\log \phi~[\rm cm^{-2}~s^{-1}]$')
        ax.set_xlim(16.8, 23.2)
        ax.set_ylim(8.8, 12.2)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=8, width=1.5)
        ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=4, width=1)
    axes[0].set_ylabel(r'$\log n_{\rm H}~[\rm cm^{-3}]$')
    handles = [Line2D([0], [0], color=col, lw=3, label=lab) for lab, col in IONS.values()]
    handles.append(Line2D([0], [0], color='0.3', lw=2.2, ls=(0, (4, 2)),
                          label=r'$\rm emission\mbox{-}weighted~\log\phi$'))
    axes[0].legend(handles=handles, loc='upper left', fontsize=14, frameon=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
