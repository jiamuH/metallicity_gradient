#!/usr/bin/env python3
"""LOC version of the composite/denominator comparison (cf. plot_nitrogen_abundance_composites.py).

Same layout as the single-zone comparison figure, but every model ratio is
LOC-integrated over the (phi, n_H) grid with Gamma = beta = -1 and with
matter-bounded clouds removed, instead of taken from one representative cloud.

The metals-3 LOC grid holds nitrogen fixed, so its R(X) curve is built as

    R_m3(X) = R_m3(1) * [ R_m1(X) / R_m1(1) ]

i.e. the normalisation comes from the metals-3 grid and the shape from the
metals-1 grid, which varies nitrogen over 0.1-9.9.  This is better than assuming
strict linearity (the ratios actually go as roughly X^0.9) and lets the same
inversion be used for every grid.

Left axis is (N/O)/(N/O)_sun, the quantity the line ratios constrain; the right
axis rescales it to N/H by the measured Z_alpha = 3, so the Z_alpha uncertainty
is a uniform shift of that axis rather than a per-point error.

Output: plots/nitrogen/Nitrogen_abundance_composites_loc.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from loc_nitrogen_abundance import (load_grid, model_ratio_curve, obs_ratio,
                                    invert, VDB, CSH, IONS)

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy',
                     'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

Z_ADOPT = 3.0
ION_COLOR = {'N3': '#5B2B7B', 'N4': '#C2185B', 'N5': '#E67E22'}
ION_LABEL = {'N3': r'$\rm N\,III]$', 'N4': r'$\rm N\,IV]$', 'N5': r'$\rm N\,V$'}

DATASETS = [
    (r'$\rm VdB01\,/\,C\,IV$',    VDB, 'C4', 'o', -0.045),
    (r'$\rm C\&S03\,/\,C\,IV$',   CSH, 'C4', 's',  0.000),
    (r'$\rm C\&S03\,/\,O\,III]$', CSH, 'O3', '^',  0.045),
]

RADII_DAT = Path('nitrogen/nitrogen_abundance_vs_r.dat')


def loc_radii():
    """Abundance-consistent LOC radii (same ensemble as the inference)."""
    from loc_nitrogen_abundance import consistent_loc_radii
    return consistent_loc_radii()


P1, F1 = load_grid('N23_m1')          # shape: nitrogen varied, metals 1
P3, F3 = load_grid('N23_m3')          # normalisation: metals 3, nitrogen fixed


def use_norm_grid(name):
    """Take the metals-3 normalisation from grid `name` (e.g. a robustness grid
    with another SED or microturbulence) instead of N23_m3."""
    global P3, F3
    P3, F3 = load_grid(name)


def curve_at_z3(ion, den):
    """R(X) at Z_alpha = 3: metals-3 normalisation, metals-1 shape."""
    xs, ys1 = model_ratio_curve(P1, F1, ion, den)
    r1_solar = float(np.interp(1.0, xs, ys1))
    _, ys3 = model_ratio_curve(P3, F3, ion, den)
    r3_solar = float(ys3[0])
    return xs, ys1 * (r3_solar / r1_solar)


def infer(ion, src, den):
    r_obs, r_err = obs_ratio(src, ion, den)
    xs, ys = curve_at_z3(ion, den)
    v, off = invert(xs, ys, r_obs)
    bounds = [invert(xs, ys, max(t, 1e-12))[0]
              for t in (r_obs - r_err, r_obs + r_err)]
    return v, min(bounds), max(bounds), off


def main(out='plots/nitrogen/Nitrogen_abundance_composites_loc.png'):
    radii = loc_radii()
    fig, ax = plt.subplots(figsize=(9.5, 7))

    for label, src, den, marker, dx in DATASETS:
        for ion in IONS:
            if ion not in src or den not in src or den not in F1:
                continue
            r = radii[ion][0] * 10 ** dx
            val, lo, hi, off = infer(ion, src, den)
            # open marker = value from power-law extrapolation beyond the grid
            ax.errorbar(r, val,
                        yerr=[[max(val - lo, 0)], [max(hi - val, 0)]],
                        fmt=marker, ms=12,
                        mfc='none' if off else ION_COLOR[ion],
                        mec=ION_COLOR[ion] if off else 'k',
                        mew=2.2 if off else 1.0, ecolor=ION_COLOR[ion],
                        elinewidth=2.0, capsize=5, zorder=4)

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
            if ion not in src or den not in src or den not in F1:
                continue
            v, lo, hi, off = infer(ion, src, den)
            print(f'{tag:>14} {ion:>5} {radii[ion][0]:>7.2f} {v:>8.2f} '
                  f'{lo:>7.2f} {hi:>7.2f} {v * Z_ADOPT:>8.2f}{" >" if off else ""}')


if __name__ == '__main__':
    main()
