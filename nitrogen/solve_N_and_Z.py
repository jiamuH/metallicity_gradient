#!/usr/bin/env python3
"""Solve for the nitrogen abundance AND the background metallicity together.

The two semi-forbidden (intercombination) nitrogen lines, N III] 1750 and
N IV] 1486, respond differently to the two quantities: raising nitrogen at fixed
background cools the gas and grows the partially ionized N^2+ zone, so N III]
responds super-linearly (dlogR/dlogX ~ 1.33) while N IV] responds sub-linearly
(~0.94).  Raising the background metallicity does something similar but not
identical.  Two lines, two unknowns, so instead of assuming Z_alpha and
inferring nitrogen from each line separately, both can be solved at once.

Model ratios are built from two one-dimensional LOC grids at N_H = 1e25,
log phi 17-21, log n 9-12 (Gamma = beta = -1):

    R(X, Z) = R(X, 1) * R(1, Z) / R(1, 1)

with R(X, 1) from the nitrogen-varied grid (N25_m1) and R(1, Z) from the
metallicity-varied grid (varym N25).  This assumes the two dependences
factorise; the (X=1, Z=1) consistency of the two independent Cloudy runs is
checked and printed, but the cross term itself is untested and is the main
caveat on this solution.

Denominators: C IV for Van den Berk (the only option there), and O III] for
Constantin & Shields (same-zone, so no C IV thermostat or C/O anomaly).

Output: plots/nitrogen/solve_N_and_Z.png  (chi^2 contours in the X-Z plane)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_aox_line_ratios import parse_line_list
from loc_nitrogen_abundance import (load_grid, model_ratio_curve, read_grd,
                                    obs_ratio, VDB, CSH, COLS, PHI_COMMON)

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy',
                     'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

VARYM = ('/Users/jiamuh/c23.01/my_models/loc_metal_flux/'
         'strong_LOC_varym_N25_v100_lineflux')
Z_ALPHA_ALPHA_ANALYSIS = 3.0        # from the alpha-element (Sec. 3) fits
OUT = Path('plots/nitrogen/solve_N_and_Z.png')


def load_varym():
    """Metallicity-varied LOC grid at X = 1: {line: flux}, plus (logphi, Z)."""
    names, rows = parse_line_list(f'{VARYM}_LineList_BLR_Fe2_flux.txt')
    params = read_grd(f'{VARYM}.grd')
    if len(params) != len(rows):
        raise RuntimeError(f'{len(params)} grid rows vs {len(rows)} data rows')
    logphi, logn, zmet = params[:, 0], params[:, 1], params[:, 2]
    keep = ((logphi >= PHI_COMMON[0]) & (logphi <= PHI_COMMON[1])
            & (logn >= 9.0))
    idx = {k: names.index(v) for k, v in COLS.items() if v in names}
    return (logphi[keep], zmet[keep],
            {k: rows[keep, i] for k, i in idx.items()})


def loc_sum(logphi, values):
    tot = 0.0
    for p in np.unique(logphi):
        m = logphi == p
        tot += np.mean(values[m]) * 10 ** (-p)      # beta = -1, Gamma = -1
    return tot


def ratio_vs_Z(logphi, zmet, flux, num, den):
    zs = np.unique(zmet)
    out = []
    for z in zs:
        m = np.isclose(zmet, z)
        out.append(loc_sum(logphi[m], flux[num][m]) / loc_sum(logphi[m], flux[den][m]))
    return zs, np.array(out)


def build_model(ion, den, gx, gz):
    """R(X, Z) on the (X, Z) mesh via the separable approximation."""
    p1, f1 = GRID_X
    xs, rx = model_ratio_curve(p1, f1, ion, den)
    rx1 = float(np.interp(1.0, xs, rx))
    lp, zm, fz = GRID_Z
    zs, rz = ratio_vs_Z(lp, zm, fz, ion, den)
    rz1 = float(np.interp(1.0, zs, rz))
    rx_i = np.interp(gx, xs, rx)
    rz_i = np.interp(gz, zs, rz)
    return rx_i * rz_i / rx1, (rx1, rz1)


def main():
    global GRID_X, GRID_Z
    GRID_X = load_grid('N25_m1')
    GRID_Z = load_varym()

    gx = np.logspace(np.log10(0.05), np.log10(9.0), 220)
    gz = np.linspace(1.0, 15.0, 200)
    GX, GZ = np.meshgrid(gx, gz, indexing='ij')

    cases = [('VdB01', VDB, 'C4'), ('C\\&S03', CSH, 'O3')]
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {'VdB01': 'dodgerblue', 'C\\&S03': 'orangered'}
    results = {}

    for label, src, den in cases:
        chi2 = np.zeros_like(GX)
        for ion in ('N3', 'N4'):
            if ion not in src:
                continue
            r_obs, r_err = obs_ratio(src, ion, den)
            model, (rx1, rz1) = build_model(ion, den, GX, GZ)
            if ion == 'N3':
                print(f'{label}/{den}: (X=1,Z=1) consistency of the two grids: '
                      f'nitrogen-grid {rx1:.4f} vs metallicity-grid {rz1:.4f} '
                      f'({rz1 / rx1:.2f}x)')
            sig = r_err / r_obs / np.log(10)          # error in dex
            chi2 += ((np.log10(model) - np.log10(r_obs)) / sig) ** 2

        i, j = np.unravel_index(np.argmin(chi2), chi2.shape)
        x_best, z_best = gx[i], gz[j]
        results[label] = (x_best, z_best, chi2.min())
        print(f'{label}: best fit  X = {x_best:.2f}  Z_alpha = {z_best:.1f}  '
              f'chi2_min = {chi2.min():.2f}  ->  N/H = {x_best * z_best:.2f}')

        d = chi2 - chi2.min()
        ax.contourf(GX, GZ, d, levels=[0, 2.30], colors=colors[label], alpha=0.30)
        ax.contour(GX, GZ, d, levels=[2.30, 6.17], colors=colors[label],
                   linewidths=2.2, linestyles=['-', '--'])
        ax.plot([x_best], [z_best], marker='*', ms=20, mfc=colors[label],
                mec='k', mew=1.0, zorder=5)

    ax.axhline(Z_ALPHA_ALPHA_ANALYSIS, color='0.35', ls=':', lw=2.2)
    ax.text(0.055, Z_ALPHA_ALPHA_ANALYSIS * 1.06,
            r'$Z_{\alpha}$ from $\alpha$-elements', color='0.35', fontsize=14)

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=colors[l], lw=2.2,
                      label=rf'$\rm {l}$') for l, _, _ in cases]
    handles += [Line2D([], [], color='0.4', lw=2.2, ls='-', label=r'$68\%$'),
                Line2D([], [], color='0.4', lw=2.2, ls='--', label=r'$95\%$')]
    ax.legend(handles=handles, fontsize=14, loc='upper right', frameon=False)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\rm (N/O)/(N/O)_{\odot}$')
    ax.set_ylabel(r'$Z_{\alpha}/Z_{\odot}$')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True,
                   length=9, width=1.5, direction='in')
    ax.tick_params(axis='both', which='minor', top=True, right=True,
                   length=4, width=1, direction='in')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
