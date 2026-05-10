#!/usr/bin/env python3
"""
Plot NIII]/CIV, NIV]/CIV, NV/CIV vs (N/H)/(N/H)_sun for the matched
Nagao "strong bump" SED reference run (single-zone, hden=10, phi(H)=19,
solar metals, vturb 100 km/s, stop column 24, no iterate convergence).

The reference run varies the nitrogen scale factor on a linear grid
(0.1..9.9 in steps of 0.2), so the curves trace the model line ratios as
a function of nitrogen abundance with all other parameters fixed.

Observed BLR composite ratios (Van den Berk et al. 2001) are overlaid as
horizontal +-1-sigma bands.

Run from repo root:
    python3 nitrogen/plot_n3n4n5_line_ratios.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from plot_aox_line_ratios import parse_line_list  # reuse parser

# Custom colormap copied from Cloudy_LOC_nitrogen_vturb.ipynb so the
# N3/N4/N5 plot uses the same palette as the original notebook.
_CUSTOM_COLORS = [
    (0.15, 0.08, 0.40),
    (0.294, 0.161, 0.569),
    (0.529, 0.173, 0.635),
    (0.753, 0.212, 0.616),
    (0.918, 0.310, 0.533),
    (0.980, 0.471, 0.463),
    (0.965, 0.663, 0.478),
    (0.929, 0.851, 0.639),
]
CUSTOM_CMAP = LinearSegmentedColormap.from_list('custom_cmap', _CUSTOM_COLORS, N=256)
COLORS_CUSTOM = [CUSTOM_CMAP(x) for x in np.linspace(0.1, 0.9, 3)]  # N3, N4, N5

plt.rcParams.update({
    'text.usetex': True,
    'axes.linewidth': 2,
    'font.family': 'serif',
    'font.weight': 'heavy',
    'font.size': 16,
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath} \boldmath'

# Multiple metallicity runs (each varying the nitrogen scale factor on the
# same linear N grid). Each is one matched Cloudy run.
MODELS = {
    'm1':  {
        'path': ('/Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100/'
                 'strong_n10_phi19_N24_m1_v100_LineList_BLR_Fe2.txt'),
        'ls':    '-',
        'label': r'$1\,Z_\odot$',
    },
    'm10': {
        'path': ('/Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100_m10/'
                 'strong_n10_phi19_N24_m10_v100_LineList_BLR_Fe2.txt'),
        'ls':    ':',
        'label': r'$10\,Z_\odot$',
    },
}
N_GRID = np.arange(0.1, 10.0, 0.2)        # nitrogen scale factor per grid block
OUT_PATH = Path('plots/nitrogen/N3N4N5_line_ratios.png')

# Cloudy line labels in the LineList header
LINE_COLS = {
    'N3': 'blnd 1750.00A',  # NIII]
    'N4': 'blnd 1486.00A',  # NIV]
    'N5': 'blnd 1240.00A',  # NV
    'C4': 'blnd 1549.00A',  # CIV (denominator)
}

# Display labels and plotting style for each line (color + LaTeX label).
# Colors match the custom palette used in the original notebook.
LINE_STYLE = [
    ('N3', r'$\rm N\,III]$',  COLORS_CUSTOM[0]),  # NIII]
    ('N4', r'$\rm N\,IV]$',   COLORS_CUSTOM[1]),  # NIV]
    ('N5', r'$\rm N\,V$',     COLORS_CUSTOM[2]),  # NV
]

# Observed BLR composite (Van den Berk et al. 2001), same values as
# nitrogen/compute_nitrogen_abundance.py.
OBS = {
    # key : (line_intensity, sigma) on the same arbitrary scale as CIV
    'N5': (2.461, 0.189),
    'N4': (0.258, 0.027),
    'N3': (0.382, 0.021),
    'C4': (25.29, 0.11),
}


def obs_ratio(num):
    """Observed line/CIV with quadrature error."""
    n, ne = OBS[num]
    d, de = OBS['C4']
    r = n / d
    re_ = r * np.sqrt((ne / n) ** 2 + (de / d) ** 2)
    return r, re_


def main():
    fig, ax = plt.subplots(figsize=(8, 6))

    # Observed +-1 sigma bands, one per line
    for key, _, color in LINE_STYLE:
        r, re_ = obs_ratio(key)
        ax.axhspan(r - re_, r + re_, color=color, alpha=0.30)

    # Model curves: loop over (metallicity, line)
    for mkey, mdef in MODELS.items():
        col_names, rows = parse_line_list(mdef['path'])
        if rows.size == 0:
            print(f'  WARNING: empty data for {mkey}; skipping')
            continue
        n_grid = N_GRID[: len(rows)]
        col_idx = {name: i for i, name in enumerate(col_names)}
        civ = rows[:, col_idx[LINE_COLS['C4']]]
        for key, _, color in LINE_STYLE:
            line_intensity = rows[:, col_idx[LINE_COLS[key]]]
            ratio = np.where(civ > 0, line_intensity / civ, np.nan)
            ax.plot(n_grid, ratio, color=color, lw=3, ls=mdef['ls'])

    # Two legends: colors -> species, linestyles -> metallicity
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
    ax.set_xlabel(r'$\rm (N/H)/(N/H)_{\odot}$')
    ax.set_ylabel(r'$\rm N_X / C\,IV\,\lambda1549$')
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
