#!/usr/bin/env python3
"""Two-panel LOC nitrogen line-ratio figure with the model spread.

Only the adopted Z_alpha = 3 LOC-averaged curve is drawn (thick solid), and the
spread over the individual (phi, n) grid cells is shown either as thin lines
(one per cell) or as a 16-84 percentile band, chosen by the style argument.
Cell ratios are rescaled by the same metals-3 normalisation factor as the LOC
curve (curve_at_z3), so cells and average share one scale.

Top panel: N_x / C IV 1549, observed bands from Vanden Berk et al. (2001).
Bottom panel: N_x / O III] 1663, bands from Constantin & Shields (2003).

The 'wband' style weights each cell by its contribution to the LOC-integrated
denominator luminosity (10^-phi x flux, i.e. the Gamma = beta = -1 weight), so
cells that barely emit the reference line do not inflate the spread.

Usage: plot_n3n4n5_loc_two_panel_spread.py [lines|band|wband]   (default: all)
Output: plots/nitrogen/N3N4N5_line_ratios_loc_2panel_{lines,band,wband}.png
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from loc_nitrogen_abundance import VDB, CSH, obs_ratio, model_ratio_curve
from plot_loc_nitrogen_composites import P1, F1, P3, F3
from plot_n3n4n5_line_ratios import LINE_STYLE   # ion colours + rcParams

OUT = 'plots/nitrogen/N3N4N5_line_ratios_loc_2panel_%s.png'


def z3_scale(ion, den):
    """Metals-3 normalisation factor applied to the metals-1 curves."""
    xs, ys1 = model_ratio_curve(P1, F1, ion, den)
    r1_solar = float(np.interp(1.0, xs, ys1))
    _, ys3 = model_ratio_curve(P3, F3, ion, den)
    return float(ys3[0]) / r1_solar


def cell_curves(ion, den):
    """(xs, ratios[n_cell, n_x], weights[n_cell, n_x]) for the (phi, n) cells.

    The weight is the cell's contribution to the LOC-integrated denominator
    luminosity, 10^-phi x flux (Gamma = beta = -1)."""
    xs = np.unique(P1['x'])
    cells = sorted(set(zip(P1['logphi'], P1['logn'])))
    out = np.full((len(cells), len(xs)), np.nan)
    wts = np.zeros((len(cells), len(xs)))
    for ic, (lp, ln_) in enumerate(cells):
        m = (P1['logphi'] == lp) & (P1['logn'] == ln_)
        xv, num, d = P1['x'][m], F1[ion][m], F1[den][m]
        order = np.argsort(xv)
        good = d[order] > 0
        r = np.full(len(xs), np.nan)
        r[good] = num[order][good] / d[order][good]
        out[ic] = r
        wts[ic] = d[order] * 10 ** (-lp)
    return xs, out, wts


def weighted_percentile(vals, wts, pcts):
    """Percentiles of vals under weights wts (both 1-D, NaNs ignored)."""
    m = np.isfinite(vals) & (wts > 0)
    if not m.any():
        return [np.nan] * len(pcts)
    order = np.argsort(vals[m])
    v, w = vals[m][order], wts[m][order]
    cum = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return [float(np.interp(p / 100.0, cum, v)) for p in pcts]


def draw_panel(ax, obs, den, style):
    for key, _, color in LINE_STYLE:
        r, re_ = obs_ratio(obs, key, den)
        ax.axhspan(r - re_, r + re_, color=color, alpha=0.30)
    for key, _, color in LINE_STYLE:
        s = z3_scale(key, den)
        xs, cells, wts = cell_curves(key, den)
        if style == 'lines':
            for row in cells:
                ax.plot(xs, row * s, color=color, lw=0.6, alpha=0.25, zorder=2)
        elif style == 'band':
            lo, hi = np.nanpercentile(cells, [16, 84], axis=0)
            ax.fill_between(xs, lo * s, hi * s, color=color, alpha=0.15,
                            lw=0, zorder=2)
        else:
            lo, hi = np.array([weighted_percentile(cells[:, j], wts[:, j],
                                                   [16, 84])
                               for j in range(len(xs))]).T
            ax.fill_between(xs, lo * s, hi * s, color=color, alpha=0.15,
                            lw=0, zorder=2)
        _, ys = model_ratio_curve(P1, {k: v for k, v in F1.items()}, key, den)
        ax.plot(xs, ys * s, color=color, lw=3, ls='-',
                solid_capstyle='round', zorder=3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True,
                   length=9, width=2, direction='in')
    ax.tick_params(axis='both', which='minor', top=True, right=True,
                   length=4, width=2, direction='in')


def make(style):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 11))
    fig.subplots_adjust(hspace=0.04)

    draw_panel(ax1, VDB, 'C4', style)
    ax1.set_ylabel(r'$\rm N_X / C\,IV\,\lambda1549$')

    draw_panel(ax2, CSH, 'O3', style)
    ax2.set_ylabel(r'$\rm N_X / O\,III]\,\lambda1663$')
    ax2.set_xlabel(r'$\rm (N/O)/(N/O)_{\odot}$')

    if style == 'lines':
        spread = Line2D([], [], color='grey', lw=0.8, alpha=0.7)
        spread_lbl = r'$\rm individual~(\phi, n)~models$'
    elif style == 'band':
        spread = Patch(color='grey', alpha=0.3)
        spread_lbl = r'$\rm 16\!-\!84\%~of~models$'
    else:
        spread = Patch(color='grey', alpha=0.3)
        spread_lbl = r'$\rm 16\!-\!84\%~of~models~(weighted)$'
    ax1.legend([Line2D([], [], color='grey', lw=3), spread],
               [r'$\rm LOC~average$', spread_lbl],
               fontsize=16, loc='upper left', frameon=False)
    ax2.legend([Line2D([], [], color=c, lw=3) for _, _, c in LINE_STYLE],
               [lbl for _, lbl, _ in LINE_STYLE],
               fontsize=16, loc='lower right', frameon=False)

    out = Path(OUT % style)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)


if __name__ == '__main__':
    styles = sys.argv[1:] or ['lines', 'band', 'wband']
    for st in styles:
        make(st)
