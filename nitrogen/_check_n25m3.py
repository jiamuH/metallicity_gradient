#!/usr/bin/env python3
"""N25 metals-3 LOC inference, using the Z = 3 slice of the alpha-element grid.

The varym N25 grid (loc_metal_flux/strong_LOC_varym_N25_v100_lineflux) varies
`metals` over 1-15 at phi 17-21, n 9-12, N_H = 1e25, nitrogen at scaled solar.
Its Z = 3 slice provides the metals-3 normalisation at N_H = 1e25; the shape of
R(X) comes from the N25_m1 vary-nitrogen grid restricted to the same phi range.
Comparing with the N23-based result tests the column-density systematic on the
metals-3 inference.

Diagnostic only; writes nothing.
"""
import numpy as np

from plot_aox_line_ratios import parse_line_list
from loc_nitrogen_abundance import (load_grid, model_ratio_curve, read_grd,
                                    obs_ratio, invert, VDB, COLS)

VARYM_DIR = '/Users/jiamuh/c23.01/my_models/loc_metal_flux'
VARYM = f'{VARYM_DIR}/strong_LOC_varym_N25_v100_lineflux'

IONS = ('N5', 'N4', 'N3')
RADII = {'N5': 2.0, 'N4': 8.987286, 'N3': 12.9539}
Z_ADOPT = 3.0


def load_varym_slice(z=3.0):
    names, rows = parse_line_list(f'{VARYM}_LineList_BLR_Fe2_flux.txt')
    params = read_grd(f'{VARYM}.grd')
    if len(params) != len(rows):
        raise RuntimeError(f'{len(params)} grid rows vs {len(rows)} data rows')
    sel = np.isclose(params[:, 2], z)
    print(f'varym N25: {len(rows)} models, {sel.sum()} at metals = {z}')
    idx = {k: names.index(v) for k, v in COLS.items() if v in names}
    flux = {k: rows[sel, i] for k, i in idx.items()}
    return dict(logphi=params[sel, 0], logn=params[sel, 1]), flux


def integrate(logphi, values):
    total = 0.0
    for p in np.unique(logphi):
        m = logphi == p
        total += np.mean(values[m]) * 10 ** (-p)     # beta = -1, Gamma = -1
    return total


def main():
    pm3, fm3 = load_varym_slice(3.0)
    p1, f1 = load_grid('N25_m1')                     # shape (phi already 17-21 common)

    print(f'\n{"ion":>5} {"model R(1) N25m3":>17} {"X":>7} {"N/H":>7}'
          f'   [N23-based: N V 1.40/4.19, N IV] 0.25/0.76, N III] 0.29/0.86]')
    lr, lz = [], []
    for ion in IONS:
        r_obs, _ = obs_ratio(VDB, ion, 'C4')
        xs, ys1 = model_ratio_curve(p1, f1, ion, 'C4')
        r1 = float(np.interp(1.0, xs, ys1))
        r3 = integrate(pm3['logphi'], fm3[ion]) / integrate(pm3['logphi'], fm3['C4'])
        ys = ys1 * (r3 / r1)
        x, off = invert(xs, ys, r_obs)
        print(f'{ion:>5} {r3:>17.4f} {x:>7.2f} {x * Z_ADOPT:>7.2f}'
              f'{"  (limit)" if off else ""}')
        lr.append(np.log10(RADII[ion]))
        lz.append(np.log10(x * Z_ADOPT))
    print(f'\nslope (polyfit through 3 points): {np.polyfit(lr, lz, 1)[0]:.2f}'
          f'   [N23-based: -0.91]')


if __name__ == '__main__':
    main()
