#!/usr/bin/env python3
"""Two-panel LOC version of the nitrogen line-ratio figure (paper Fig. 2).

LOC-averaged model ratios (beta = Gamma = -1, common window) as functions of
the nitrogen abundance X = (N/O)/(N/O)_sun.  The solid curve is the adopted
Z_alpha = 3 model actually used for the inference (curve_at_z3: metals-3
normalisation, metals-1 shape); the Z_alpha = 1 and 10 curves apply the
separable-model metallicity factor, which shifts each ratio vertically.

Top panel: N_x / C IV 1549; bottom panel: N_x / O III] 1663. Observed bands in
both panels are from Constantin & Shields (2003).

Output: plots/nitrogen/N3N4N5_line_ratios_loc_2panel.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from loc_nitrogen_abundance import CSH, obs_ratio
from plot_loc_nitrogen_composites import curve_at_z3
from solve_N_and_Z import load_varym, ratio_vs_Z
from plot_n3n4n5_line_ratios import LINE_STYLE   # ion colours + rcParams

Z_ADOPT = 3.0
# (Z_alpha, linestyle); solid = adopted
Z_CURVES = [(Z_ADOPT, '-'), (10.0, ':')]

OUT = Path('plots/nitrogen/N3N4N5_line_ratios_loc_2panel.png')

_GRID_Z = load_varym()


def z_factor(ion, den, z):
    """Separable-model vertical shift R(X, z) / R(X, Z_ADOPT)."""
    lp, zm, fz = _GRID_Z
    zs, rz = ratio_vs_Z(lp, zm, fz, ion, den)
    return float(np.interp(z, zs, rz) / np.interp(Z_ADOPT, zs, rz))


def draw_panel(ax, obs, den):
    for key, _, color in LINE_STYLE:
        r, re_ = obs_ratio(obs, key, den)
        ax.axhspan(r - re_, r + re_, color=color, alpha=0.30)
    for key, _, color in LINE_STYLE:
        xs, ys = curve_at_z3(key, den)
        for z, ls in Z_CURVES:
            ax.plot(xs, ys * z_factor(key, den, z), color=color, lw=3, ls=ls,
                    solid_capstyle='round', dash_capstyle='round')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True,
                   length=9, width=2, direction='in')
    ax.tick_params(axis='both', which='minor', top=True, right=True,
                   length=4, width=2, direction='in')


def main():
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 11))
    fig.subplots_adjust(hspace=0.04)

    draw_panel(ax1, CSH, 'C4')
    ax1.set_ylabel(r'$\rm N_X / C\,IV\,\lambda1549$')

    draw_panel(ax2, CSH, 'O3')
    ax2.set_ylabel(r'$\rm N_X / O\,III]\,\lambda1663$')
    ax2.set_xlabel(r'$\rm (N/O)/(N/O)_{\odot}$')

    style_handles = [Line2D([], [], color='grey', lw=3, ls=ls)
                     for _, ls in Z_CURVES]
    style_labels = [r'$Z_{\alpha} = %g~Z_{\odot}$' % z for z, _ in Z_CURVES]
    ax1.legend(style_handles, style_labels, fontsize=16, loc='upper left',
               frameon=False)
    color_handles = [Line2D([], [], color=color, lw=3)
                     for _, _, color in LINE_STYLE]
    color_labels = [lbl for _, lbl, _ in LINE_STYLE]
    ax2.legend(color_handles, color_labels, fontsize=16, loc='lower right',
               frameon=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    print(f'Saved: {OUT}')
    plt.close(fig)


if __name__ == '__main__':
    main()
