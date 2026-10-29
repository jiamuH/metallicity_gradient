#!/usr/bin/env python3
"""Nitrogen abundance vs BLR radius: N/O measured, N/H via the measured Z_alpha.

The Cloudy grid variable X (`element nitrogen scale factor`) multiplies on top of
the `metals` scaling, so what a line ratio constrains is

    X = (N/O)/(N/O)_sun ,   and   (N/H)/(N/H)_sun = X * Z_alpha/Zsun .

Rather than averaging the Z_alpha = 1 and 10 Zsun models (whose N/H differ by a
factor 10 by construction), each ion's N/O is interpolated in log Z_alpha to the
metallicity measured from the alpha-element analysis, Z_alpha = 3 Zsun, and the
right-hand axis rescales that to N/H by the same constant factor.  The Z_alpha
uncertainty is therefore a uniform vertical shift of the right axis, not a
per-point error.

Three (composite, denominator) combinations are compared:
  Van den Berk 2001 / C IV                 (general quasar composite)
  Constantin & Shields 2003 / C IV
  Constantin & Shields 2003 / O III] 1663  (same-zone intercombination pair)

Output: plots/nitrogen/Nitrogen_abundance_composites.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from compute_nitrogen_abundance import MODEL_FILES, N_GRID
from plot_aox_line_ratios import parse_line_list

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy',
                     'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

COLS = {'N5': 'blnd 1240.00A', 'N4': 'blnd 1486.00A', 'N3': 'blnd 1750.00A',
        'C4': 'blnd 1549.00A', 'O3': 'blnd 1666.00A'}
Z_ALPHA = {'m1': 1.0, 'm10': 10.0}
Z_ADOPT = 3.0            # Z_alpha measured in the alpha-element analysis
IONS = ('N5', 'N4', 'N3')

ION_COLOR = {'N3': '#5B2B7B', 'N4': '#C2185B', 'N5': '#E67E22'}
ION_LABEL = {'N3': r'$\rm N\,III]$', 'N4': r'$\rm N\,IV]$', 'N5': r'$\rm N\,V$'}

VDB = {'N5': (2.461, 0.189), 'N4': (0.258, 0.027), 'N3': (0.382, 0.021),
       'C4': (25.29, 0.11)}
CSH = {'N5': (13.0, 1.5), 'N4': (0.7, 0.3), 'N3': (2.0, 0.2),
       'C4': (44.8, 1.7), 'O3': (4.1, 1.1)}

DATASETS = [
    (r'$\rm VdB01\,/\,C\,IV$',    VDB, 'C4', 'o', -0.045),
    (r'$\rm C\&S03\,/\,C\,IV$',   CSH, 'C4', 's',  0.000),
    (r'$\rm C\&S03\,/\,O\,III]$', CSH, 'O3', '^',  0.045),
]

RADII_DAT = Path('nitrogen/nitrogen_abundance_vs_r.dat')


def loc_radii():
    """{ion: (r, r_lo, r_hi)} in light-days, from the published .dat.

    LOC emission-weighted radii of each ion's own radial profile; independent of
    the denominator used for the abundance inference.  (loc_emission_weighted_r.py
    recomputes these but requires numpy >= 2.0.)
    """
    out = {}
    for line in RADII_DAT.read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        p = line.split()
        out[p[0]] = (float(p[1]), float(p[2]), float(p[3]))
    return out


def load(path):
    names, rows = parse_line_list(path)
    idx = {k: names.index(v) for k, v in COLS.items()}
    return {k: rows[:, i] for k, i in idx.items()}


GRIDS = {m: load(p) for m, p in MODEL_FILES.items()}


def obs_ratio(src, num, den):
    n, ne = src[num]
    d, de = src[den]
    r = n / d
    return r, r * np.hypot(ne / n, de / d)


def invert(x, y, target):
    """(N/O, off_grid_flag) where the model ratio curve equals target."""
    order = np.argsort(y)
    ys, xs = y[order], x[order]
    if target > ys[-1]:
        return float(xs[-1]), True
    if target < ys[0]:
        return float(xs[0]), False
    return float(np.interp(target, ys, xs)), False


_W = np.log10(Z_ADOPT) / np.log10(Z_ALPHA['m10'])     # interpolation weight


def no_at_zadopt(ion, den, target):
    """N/O at Z_alpha = Z_ADOPT, log-interpolated between the two model runs."""
    vals, off = {}, False
    for mkey, d in GRIDS.items():
        n = min(len(d[ion]), len(N_GRID))
        v, flag = invert(N_GRID[:n], (d[ion] / d[den])[:n], target)
        vals[mkey] = v
        off |= flag
    lo, hi = np.log10(vals['m1']), np.log10(vals['m10'])
    return 10 ** (lo + _W * (hi - lo)), off


def infer(ion, src, den):
    """(N/O best, lo, hi, off_grid) at Z_alpha = Z_ADOPT."""
    r_obs, r_err = obs_ratio(src, ion, den)
    best, off = no_at_zadopt(ion, den, r_obs)
    bounds = [no_at_zadopt(ion, den, max(t, 1e-8))[0]
              for t in (r_obs - r_err, r_obs + r_err)]
    return best, min(bounds), max(bounds), off


def main(out='plots/nitrogen/Nitrogen_abundance_composites.png'):
    radii = loc_radii()
    fig, ax = plt.subplots(figsize=(9.5, 7))

    for label, src, den, marker, dx in DATASETS:
        for ion in IONS:
            if ion not in src or den not in src:
                continue
            r = radii[ion][0] * 10 ** dx
            val, lo, hi, off = infer(ion, src, den)
            if off:
                ax.errorbar(r, val, yerr=[[0.35 * val], [0.0]], lolims=True,
                            fmt=marker, ms=12, mfc='none', mec=ION_COLOR[ion],
                            mew=2.2, ecolor=ION_COLOR[ion], elinewidth=2.0,
                            capsize=5, zorder=4)
            else:
                ax.errorbar(r, val,
                            yerr=[[max(val - lo, 0)], [max(hi - val, 0)]],
                            fmt=marker, ms=12, mfc=ION_COLOR[ion], mec='k',
                            mew=1.0, ecolor=ION_COLOR[ion], elinewidth=2.0,
                            capsize=5, zorder=4)

    ax.axhline(1.0, color='0.5', ls=':', lw=1.8, zorder=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.06, 30)
    ax.set_xlabel(r'$r_{\rm BLR}~[\rm ld]$')
    ax.set_ylabel(r'$\rm (N/\alpha)/(N/\alpha)_{\odot}$')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=False,
                   length=10, width=1.5, direction='in')
    ax.tick_params(axis='both', which='minor', top=True, right=False,
                   length=5, width=1, direction='in')

    # right axis: same numbers rescaled by the measured Z_alpha
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(ax.get_ylim()[0] * Z_ADOPT, ax.get_ylim()[1] * Z_ADOPT)
    ax2.set_ylabel(r'$\rm (N/H)/(N/H)_{\odot}~at~\it{Z_\alpha}\rm =3\,\it{Z_\odot}$',
                   rotation=270, labelpad=30)
    ax2.minorticks_on()
    ax2.tick_params(axis='y', which='major', length=10, width=1.5, direction='in')
    ax2.tick_params(axis='y', which='minor', length=5, width=1, direction='in')

    ion_handles = [Line2D([], [], marker='o', ls='', ms=11, mfc=ION_COLOR[k],
                          mec='k', label=ION_LABEL[k]) for k in IONS]
    set_handles = [Line2D([], [], marker=m, ls='', ms=11, mfc='0.6', mec='k',
                          label=lab) for lab, _, _, m, _ in DATASETS]
    leg1 = ax.legend(handles=ion_handles, loc='upper right', fontsize=14,
                     frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=set_handles, loc='lower left', fontsize=14, frameon=False)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_extra_artists=[ax2.yaxis.label])
    plt.close(fig)
    print(f'Saved {out}')

    print(f'\n{"set":>14} {"ion":>5} {"r[ld]":>7} {"N/O":>8} {"lo":>7} {"hi":>7} '
          f'{"N/H":>8}')
    for label, src, den, _, _ in DATASETS:
        tag = ('VdB' if src is VDB else 'C&S') + '/' + den
        for ion in IONS:
            if ion not in src or den not in src:
                continue
            v, lo, hi, off = infer(ion, src, den)
            mark = ' >' if off else '  '
            print(f'{tag:>14} {ion:>5} {radii[ion][0]:>7.2f} {v:>8.2f} '
                  f'{lo:>7.2f} {hi:>7.2f} {v * Z_ADOPT:>8.2f}{mark}')


if __name__ == '__main__':
    main()
