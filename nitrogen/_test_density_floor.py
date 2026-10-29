#!/usr/bin/env python3
"""Does a density floor in the LOC average flip the N III]/N IV] ordering?

N IV] 1486 (n_crit ~ 5e9) emits efficiently in the lowest-density clouds of the
grid; N III] 1750 (n_crit ~ 1.5e10) less so.  Restricting the LOC density range
to n >= n_min suppresses the model N IV] more than N III], raising the inferred
X_N4 relative to X_N3.  This sweep quantifies that (VdB01 / C IV, beta = -1,
metals-3 normalisation with metals-1 shape, baseline Stromgren cut).

Diagnostic only; writes nothing.
"""
import numpy as np

from loc_nitrogen_abundance import load_grid, obs_ratio, invert, VDB

IONS = ('N5', 'N4', 'N3')
RADII = {'N5': 2.0, 'N4': 8.987286, 'N3': 12.9539}
Z_ADOPT = 3.0


def integrate(logphi, logn, values, n_min):
    total = 0.0
    for p in np.unique(logphi):
        m = (logphi == p) & (logn >= n_min)
        if not m.any():
            continue
        total += np.mean(values[m]) * 10 ** (-p)     # beta = -1
    return total


def curve(params, flux, num, den, n_min):
    xs = np.unique(params['x'])
    ys = []
    for xv in xs:
        m = params['x'] == xv
        n = integrate(params['logphi'][m], params['logn'][m], flux[num][m], n_min)
        d = integrate(params['logphi'][m], params['logn'][m], flux[den][m], n_min)
        ys.append(n / d)
    return xs, np.array(ys)


def main():
    p1, f1 = load_grid('N23_m1')
    p3, f3 = load_grid('N23_m3')
    print(f'{"n_min":>7} {"N V":>7} {"N IV]":>7} {"N III]":>7} '
          f'{"N4/N3":>7} {"slope":>7}')
    for n_min in (9.0, 9.5, 10.0, 10.5):
        nh = {}
        for ion in IONS:
            r_obs, _ = obs_ratio(VDB, ion, 'C4')
            xs, ys1 = curve(p1, f1, ion, 'C4', n_min)
            r1 = float(np.interp(1.0, xs, ys1))
            _, ys3 = curve(p3, f3, ion, 'C4', n_min)
            ys = ys1 * (float(ys3[0]) / r1)
            nh[ion] = invert(xs, ys, r_obs)[0] * Z_ADOPT
        lr = np.log10([RADII[i] for i in IONS])
        lz = np.log10([nh[i] for i in IONS])
        slope = np.polyfit(lr, lz, 1)[0]
        print(f'{n_min:>7.1f} {nh["N5"]:>7.2f} {nh["N4"]:>7.2f} '
              f'{nh["N3"]:>7.2f} {nh["N4"] / nh["N3"]:>7.2f} {slope:>7.2f}')


if __name__ == '__main__':
    main()
