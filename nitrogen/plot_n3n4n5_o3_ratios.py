#!/usr/bin/env python3
"""O III]-denominator companion of Figure 2: N_x / O III] 1663 vs (N/O)/(N/O)_sun.

Same matched single-zone Cloudy models and layout as plot_n3n4n5_line_ratios.py,
but with O III] 1663 as the denominator and the observed bands from the
Constantin & Shields (2003) NLS1 composite (Table 3), the only composite in use
here that reports O III] 1663.

Output: plots/nitrogen/N3N4N5_O3_line_ratios.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_aox_line_ratios import parse_line_list
from plot_n3n4n5_line_ratios import (MODELS, N_GRID, LINE_STYLE, plt as _plt)  # noqa: F401
# importing plot_n3n4n5_line_ratios also applies the shared rcParams

LINE_COLS = {
    'N3': 'blnd 1750.00A',
    'N4': 'blnd 1486.00A',
    'N5': 'blnd 1240.00A',
    'O3': 'blnd 1666.00A',   # O III] 1663 denominator
}

# Constantin & Shields (2003) Table 3, 100 F/F(Lya)
OBS = {
    'N5': (13.0, 1.5),
    'N4': (0.7, 0.3),
    'N3': (2.0, 0.2),
    'O3': (4.1, 1.1),
}

OUT_PATH = Path('plots/nitrogen/N3N4N5_O3_line_ratios.png')


def obs_ratio(num):
    n, ne = OBS[num]
    d, de = OBS['O3']
    r = n / d
    return r, r * np.sqrt((ne / n) ** 2 + (de / d) ** 2)


def main():
    fig, ax = plt.subplots(figsize=(8, 6))

    for key, _, color in LINE_STYLE:
        r, re_ = obs_ratio(key)
        ax.axhspan(r - re_, r + re_, color=color, alpha=0.30)

    for mkey, mdef in MODELS.items():
        col_names, rows = parse_line_list(mdef['path'])
        if rows.size == 0:
            print(f'  WARNING: empty data for {mkey}; skipping')
            continue
        n_grid = N_GRID[: len(rows)]
        col_idx = {name: i for i, name in enumerate(col_names)}
        o3 = rows[:, col_idx[LINE_COLS['O3']]]
        for key, _, color in LINE_STYLE:
            line_intensity = rows[:, col_idx[LINE_COLS[key]]]
            ratio = np.where(o3 > 0, line_intensity / o3, np.nan)
            ax.plot(n_grid, ratio, color=color, lw=3, ls=mdef['ls'])

    color_handles = [Line2D([], [], color=color, lw=3) for _, _, color in LINE_STYLE]
    color_labels = [lbl for _, lbl, _ in LINE_STYLE]
    style_handles = [Line2D([], [], color='grey', lw=3, ls=m['ls'])
                     for m in MODELS.values()]
    style_labels = [m['label'] for m in MODELS.values()]
    leg_color = ax.legend(color_handles, color_labels, fontsize=16,
                          loc='lower right', frameon=False)
    ax.add_artist(leg_color)
    ax.legend(style_handles, style_labels, fontsize=16, loc='upper left',
              frameon=False)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\rm (N/O)/(N/O)_{\odot}$')
    ax.set_ylabel(r'$\rm N_X / O\,III]\,\lambda1663$')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True,
                   length=9, width=2, direction='in')
    ax.tick_params(axis='both', which='minor', top=True, right=True,
                   length=4, width=2, direction='in')

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=300, bbox_inches='tight')
    print(f'Saved: {OUT_PATH}')
    plt.close(fig)


if __name__ == '__main__':
    main()
