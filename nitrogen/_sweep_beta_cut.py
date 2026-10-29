#!/usr/bin/env python3
"""Systematics sweep: LOC density weight beta x matter-bounded cut placement.

For each combination, redo the nitrogen inference (VdB01 / C IV, metals-3
normalisation with metals-1 shape, as in the LOC figures) and report the three
ion abundances plus the power-law slope through the three (r, N/H) points.

Cut variants (offsets to the Stromgren threshold log phi - log n < log(alpha_B N_H)):
    -0.3 dex : stricter (ionization front must sit within half the slab)
     0.0 dex : baseline
    +0.3 dex : looser (allows fronts up to twice the column)
     off     : no matter-bounded cut at all

Diagnostic only; writes nothing.
"""
import numpy as np

from loc_nitrogen_abundance import load_grid, obs_ratio, invert, VDB

IONS = ('N5', 'N4', 'N3')
RADII = {'N5': 2.0, 'N4': 8.987286, 'N3': 12.9539}   # light-days (published)
Z_ADOPT = 3.0
BETAS = (-1.5, -1.0, -0.5)
CUTS = ((-0.3, ' -0.3'), (0.0, '  0.0'), (0.3, ' +0.3'), (np.inf, '  off'))


def integrate(logphi, logn, values, beta):
    total = 0.0
    for p in np.unique(logphi):
        m = logphi == p
        w = 10 ** ((beta + 1.0) * logn[m])
        total += np.sum(values[m] * w) / np.sum(w) * 10 ** (-p)
    return total


def curve(params, flux, num, den, beta):
    xs = np.unique(params['x'])
    ys = []
    for xv in xs:
        m = params['x'] == xv
        n = integrate(params['logphi'][m], params['logn'][m], flux[num][m], beta)
        d = integrate(params['logphi'][m], params['logn'][m], flux[den][m], beta)
        ys.append(n / d)
    return xs, np.array(ys)


def main():
    print(f'{"beta":>6} {"cut":>6} {"kept m3":>8} {"N V":>7} {"N IV]":>7} '
          f'{"N III]":>7} {"slope":>7}')
    for off, cut_lab in CUTS:
        p1, f1 = load_grid('N23_m1', strom_offset=off)
        p3, f3 = load_grid('N23_m3', strom_offset=off)
        n_kept = int(len(p3['logphi']))
        for beta in BETAS:
            nh = {}
            for ion in IONS:
                r_obs, _ = obs_ratio(VDB, ion, 'C4')
                xs, ys1 = curve(p1, f1, ion, 'C4', beta)
                r1_solar = float(np.interp(1.0, xs, ys1))
                _, ys3 = curve(p3, f3, ion, 'C4', beta)
                ys = ys1 * (float(ys3[0]) / r1_solar)
                nh[ion] = invert(xs, ys, r_obs)[0] * Z_ADOPT
            lr = np.log10([RADII[i] for i in IONS])
            lz = np.log10([nh[i] for i in IONS])
            slope = np.polyfit(lr, lz, 1)[0]
            print(f'{beta:>6.1f} {cut_lab:>6} {n_kept:>8d} {nh["N5"]:>7.2f} '
                  f'{nh["N4"]:>7.2f} {nh["N3"]:>7.2f} {slope:>7.2f}')


if __name__ == '__main__':
    main()
