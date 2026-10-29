#!/usr/bin/env python3
"""Why does N III] imply a larger nitrogen abundance than N IV]?

Three checks on the LOC grids:

1. Emission-weighted mean (log n_H, log phi) of each line, i.e. which clouds
   dominate the LOC-integrated luminosity.  If N III] is produced by systematically
   lower-density or lower-ionization gas than N IV] and C IV, then the inferred
   abundance difference is a statement about the assumed cloud distribution rather
   than about nitrogen.

2. Sensitivity of the inferred nitrogen scale factor to the LOC density weight
   beta (the canonical value is -1).  A line near its critical density responds
   strongly to beta.

3. Observed vs model N III]/N IV], which separates an observational cause (the
   composites disagreeing about the two line strengths) from a modelling one.

Diagnostic only; writes nothing.
"""
import numpy as np

from loc_nitrogen_abundance import (load_grid, model_ratio_curve, obs_ratio,
                                    invert, VDB, CSH)

LINES = ('N3', 'N4', 'N5', 'C4', 'O3')
LABEL = {'N3': 'N III] 1750', 'N4': 'N IV] 1486', 'N5': 'N V 1240',
         'C4': 'C IV 1549', 'O3': 'O III] 1663'}


def weighted_means(params, flux, beta=-1.0):
    """Emission-weighted <log n> and <log phi> per line for the LOC integrand."""
    lp, ln = params['logphi'], params['logn']
    at_solar = np.isclose(params['x'], params['x'][np.argmin(np.abs(params['x'] - 1.0))])
    out = {}
    for key in LINES:
        if key not in flux:
            continue
        f = flux[key][at_solar]
        lpm, lnm = lp[at_solar], ln[at_solar]
        # LOC weight: radial r^(Gamma+3) ~ phi^-1, density n^(beta+1) per dlog n
        w = f * 10 ** (-lpm) * 10 ** ((beta + 1.0) * lnm)
        out[key] = (float(np.sum(w * lnm) / np.sum(w)),
                    float(np.sum(w * lpm) / np.sum(w)),
                    float(np.sum(w * (lpm - lnm)) / np.sum(w)))
    return out


def loc_integrate_beta(logphi, logn, values, beta):
    total = 0.0
    for p in np.unique(logphi):
        m = logphi == p
        w = 10 ** ((beta + 1.0) * logn[m])
        total += np.sum(values[m] * w) / np.sum(w) * 10 ** (-p)
    return total


def ratio_curve_beta(params, flux, num, den, beta):
    xs = np.unique(params['x'])
    ys = []
    for xv in xs:
        m = params['x'] == xv
        n = loc_integrate_beta(params['logphi'][m], params['logn'][m], flux[num][m], beta)
        d = loc_integrate_beta(params['logphi'][m], params['logn'][m], flux[den][m], beta)
        ys.append(n / d)
    return xs, np.array(ys)


def main():
    p1, f1 = load_grid('N23_m1')
    p3, f3 = load_grid('N23_m3')

    print('\n1. Emission-weighted cloud conditions (metals 3, solar N, beta = -1)')
    print(f'   {"line":>14} {"<log n>":>9} {"<log phi>":>10} {"<log phi - log n>":>18}')
    for key, (mn, mp, mu) in weighted_means(p3, f3).items():
        print(f'   {LABEL[key]:>14} {mn:>9.2f} {mp:>10.2f} {mu:>18.2f}')

    print('\n2. Inferred X vs the LOC density weight beta (metals 1, VdB01/C IV)')
    print(f'   {"beta":>6} {"N III]":>9} {"N IV]":>9} {"N V":>9} {"N3/N4":>8}')
    for beta in (-2.0, -1.5, -1.0, -0.5, 0.0):
        vals = {}
        for ion in ('N3', 'N4', 'N5'):
            r_obs, _ = obs_ratio(VDB, ion, 'C4')
            xs, ys = ratio_curve_beta(p1, f1, ion, 'C4', beta)
            vals[ion] = invert(xs, ys, r_obs)[0]
        print(f'   {beta:>6.1f} {vals["N3"]:>9.2f} {vals["N4"]:>9.2f} '
              f'{vals["N5"]:>9.2f} {vals["N3"] / vals["N4"]:>8.2f}')

    print('\n3. Observed vs model N III]/N IV]')
    for name, src in (('VdB01', VDB), ('C&S03', CSH)):
        r = src['N3'][0] / src['N4'][0]
        print(f'   observed {name}: {r:.2f}')
    for tag, params, flux in (('metals 1', p1, f1), ('metals 3', p3, f3)):
        xs, ys = model_ratio_curve(params, flux, 'N3', 'N4')
        r_solar = float(np.interp(1.0, xs, ys)) if len(xs) > 1 else float(ys[0])
        print(f'   model {tag} (solar N/alpha): {r_solar:.2f}')


if __name__ == '__main__':
    main()
