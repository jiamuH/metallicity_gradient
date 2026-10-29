#!/usr/bin/env python3
"""Two-panel version of the nitrogen line-ratio figure (paper Fig. 2).

Top panel: N_x / C IV 1549 with the Van den Berk et al. (2001) observed bands
(as in the original single-panel figure).  Bottom panel: N_x / O III] 1663 with
the Constantin & Shields (2003) bands (the only composite in hand with an
O III] measurement).  Both panels share the (N/O)/(N/O)_sun axis and show the
same matched single-zone Cloudy models at 1 and 10 Zsun.

Output: plots/nitrogen/N3N4N5_line_ratios_2panel.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_aox_line_ratios import parse_line_list
from plot_n3n4n5_line_ratios import (MODELS, N_GRID, LINE_STYLE,
                                     OBS as OBS_VDB)   # applies shared rcParams
from plot_n3n4n5_o3_ratios import OBS as OBS_CSH

LINE_COLS = {
    'N3': 'blnd 1750.00A',
    'N4': 'blnd 1486.00A',
    'N5': 'blnd 1240.00A',
    'C4': 'blnd 1549.00A',
    'O3': 'blnd 1666.00A',
}

OUT = Path('plots/nitrogen/N3N4N5_line_ratios_2panel.png')


def obs_ratio(obs, num, den):
    n, ne = obs[num]
    d, de = obs[den]
    r = n / d
    return r, r * np.sqrt((ne / n) ** 2 + (de / d) ** 2)


def draw_panel(ax, den, obs, den_key):
    for key, _, color in LINE_STYLE:
        r, re_ = obs_ratio(obs, key, den_key)
        ax.axhspan(r - re_, r + re_, color=color, alpha=0.30)
    for mkey, mdef in MODELS.items():
        col_names, rows = parse_line_list(mdef['path'])
        if rows.size == 0:
            continue
        n_grid = N_GRID[: len(rows)]
        idx = {name: i for i, name in enumerate(col_names)}
        den_flux = rows[:, idx[LINE_COLS[den]]]
        for key, _, color in LINE_STYLE:
            ratio = np.where(den_flux > 0,
                             rows[:, idx[LINE_COLS[key]]] / den_flux, np.nan)
            ax.plot(n_grid, ratio, color=color, lw=3, ls=mdef['ls'],
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

    draw_panel(ax1, 'C4', OBS_VDB, 'C4')
    ax1.set_ylabel(r'$\rm N_X / C\,IV\,\lambda1549$')

    draw_panel(ax2, 'O3', OBS_CSH, 'O3')
    ax2.set_ylabel(r'$\rm N_X / O\,III]\,\lambda1663$')
    ax2.set_xlabel(r'$\rm (N/O)/(N/O)_{\odot}$')

    # legends: metallicity line styles (top panel), ion colours (bottom panel)
    style_handles = [Line2D([], [], color='grey', lw=3, ls=m['ls'])
                     for m in MODELS.values()]
    style_labels = [m['label'] for m in MODELS.values()]
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
