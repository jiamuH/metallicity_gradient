#!/usr/bin/env python3
"""Model map of the same-zone ratio N III]/N IV] over (Z_alpha, density floor).

N III] and N IV] form in the same clouds, so the ratio of their inferred
abundances equals observed(N III]/N IV]) / model(N III]/N IV]).  This map shows
which (Z_alpha, n_min) combinations raise the model ratio enough to remove the
apparent N III]-N IV] abundance inversion.

Grid: the varym N25 alpha-element grid (metals 1-15, phi 17-21, n 9-12,
N_H = 1e25, nitrogen scaled-solar), LOC-averaged with Gamma = beta = -1 and a
variable density floor n_min.  Note the baseline nitrogen analysis uses
N_H = 1e23, where the model ratio at Z = 3, n_min = 9 is 1.09 rather than the
~1.4 of this N25 grid; the map is read as a trend, with contours marking the
observed ratios and the gradient-consistent targets.

Contours:
  1.48  observed VdB01
  2.86  observed C&S03
  3.10  model ratio needed for VdB01 to be consistent with k_N = -1.27
        between the N IV] and N III] radii (4.83 -> 8.63 ld)

Output: plots/nitrogen/model_n3n4_map.png
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_aox_line_ratios import parse_line_list
from loc_nitrogen_abundance import read_grd

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy',
                     'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

VARYM = ('/Users/jiamuh/c23.01/my_models/loc_metal_flux/'
         'strong_LOC_varym_N25_v100_lineflux')
COLS = {'N3': 'blnd 1750.00A', 'N4': 'blnd 1486.00A'}

OUT = Path('plots/nitrogen/model_n3n4_map.png')


def main():
    names, rows = parse_line_list(f'{VARYM}_LineList_BLR_Fe2_flux.txt')
    params = read_grd(f'{VARYM}.grd')
    if len(params) != len(rows):
        raise RuntimeError(f'{len(params)} grid rows vs {len(rows)} data rows')
    logphi, logn, zmet = params[:, 0], params[:, 1], params[:, 2]
    idx = {k: names.index(v) for k, v in COLS.items()}
    f3, f4 = rows[:, idx['N3']], rows[:, idx['N4']]

    z_ax = np.unique(zmet)
    nmin_ax = np.arange(9.0, 11.6, 0.25)
    ratio = np.full((len(nmin_ax), len(z_ax)), np.nan)
    for iz, z in enumerate(z_ax):
        for i_n, n_min in enumerate(nmin_ax):
            sel = np.isclose(zmet, z) & (logn >= n_min - 1e-6)
            num = den = 0.0
            for p in np.unique(logphi[sel]):
                m = sel & (logphi == p)
                num += np.mean(f3[m]) * 10 ** (-p)      # beta = -1, Gamma = -1
                den += np.mean(f4[m]) * 10 ** (-p)
            if den > 0:
                ratio[i_n, iz] = num / den

    fig, ax = plt.subplots(figsize=(9, 7))
    pc = ax.pcolormesh(z_ax, nmin_ax, np.log10(ratio), cmap='viridis',
                       shading='nearest')
    cbar = fig.colorbar(pc, ax=ax, pad=0.02)
    cbar.set_label(r'$\log\,({\rm N\,III]/N\,IV]})_{\rm model}$',
                   rotation=270, labelpad=24)
    cbar.ax.tick_params(direction='in')

    levels = [1.48, 2.86, 3.10]
    styles = ['-', '--', ':']
    labels = [r'$\rm obs.~VdB01~(1.48)$', r'$\rm obs.~C\&S03~(2.86)$',
              r'$\rm VdB01 + \it{k}_{\rm N}\rm ~consistent~(3.1)$']
    cs = ax.contour(z_ax, nmin_ax, ratio, levels=levels, colors='white',
                    linestyles=styles, linewidths=2.5)
    handles = [plt.Line2D([], [], color='white', ls=s, lw=2.5) for s in styles]
    leg = ax.legend(handles, labels, fontsize=13, loc='lower right',
                    frameon=True, facecolor='0.25', edgecolor='none',
                    labelcolor='white')

    ax.plot([3.0], [9.0], marker='*', ms=20, mfc='orangered', mec='white',
            mew=1.5, zorder=5)
    ax.text(3.25, 9.06, r'$\rm baseline$', color='white', fontsize=14)

    ax.set_xlabel(r'$Z_{\alpha}/Z_{\odot}$')
    ax.set_ylabel(r'$\log\,n_{\rm min}~[\rm cm^{-3}]$')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True,
                   length=8, width=1.5, direction='in', color='white')
    ax.tick_params(axis='both', which='minor', top=True, right=True,
                   length=4, width=1, direction='in', color='white')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f'Saved {OUT}')

    # console summary along n_min = 9 and Z = 3
    iz3 = int(np.argmin(np.abs(z_ax - 3.0)))
    print('\nmodel N3/N4 at n_min = 9 vs Z:',
          ', '.join(f'Z={z:.0f}: {ratio[0, i]:.2f}'
                    for i, z in enumerate(z_ax) if z in (1, 3, 5, 10, 15)))
    print('model N3/N4 at Z = 3 vs n_min:',
          ', '.join(f'{n:.2f}: {ratio[i, iz3]:.2f}'
                    for i, n in enumerate(nmin_ax) if n % 0.5 == 0))


if __name__ == '__main__':
    main()
