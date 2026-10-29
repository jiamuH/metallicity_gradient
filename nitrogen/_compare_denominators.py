#!/usr/bin/env python3
"""Compare nitrogen abundances inferred with C IV vs O III] as the denominator.

The Cloudy grid variable X (the `element nitrogen scale factor`) multiplies on
top of the `metals` scaling, so

    n(N)/n(H) = X * (Z_alpha/Zsun) * [n(N)/n(H)]_sun,

i.e. X is (N/O)/(N/O)_sun, nitrogen relative to the alpha elements.  N/H
relative to solar is therefore X * Z_alpha.

Prints, for each nitrogen ion and each Z_alpha model, the inferred N/O and the
implied N/H, for both denominators.  Diagnostic only; writes nothing.
"""
import numpy as np

from compute_nitrogen_abundance import MODEL_FILES, N_GRID
from plot_aox_line_ratios import parse_line_list

COLS = {'N5': 'blnd 1240.00A', 'N4': 'blnd 1486.00A', 'N3': 'blnd 1750.00A',
        'C4': 'blnd 1549.00A', 'O3': 'blnd 1666.00A'}

Z_ALPHA = {'m1': 1.0, 'm10': 10.0}

# Van den Berk et al. (2001) composite, arbitrary common scale (as used in
# compute_nitrogen_abundance.py).
VDB = {'N5': (2.461, 0.189), 'N4': (0.258, 0.027), 'N3': (0.382, 0.021),
       'C4': (25.29, 0.11)}

# Constantin & Shields (2003) NLS1 composite, Table 3, 100*F/F(Lya).
CSH = {'N5': (13.0, 1.5), 'N4': (0.7, 0.3), 'N3': (2.0, 0.2),
       'C4': (44.8, 1.7), 'O3': (4.1, 1.1)}


def load(path):
    names, rows = parse_line_list(path)
    idx = {k: names.index(v) for k, v in COLS.items()}
    return {k: rows[:, i] for k, i in idx.items()}


def obs_ratio(src, num, den):
    n, ne = src[num]
    d, de = src[den]
    r = n / d
    return r, r * np.hypot(ne / n, de / d)


def invert(x, y, target):
    """X where the model ratio curve y(x) equals target (monotonic rising)."""
    order = np.argsort(y)
    if target < y[order][0] or target > y[order][-1]:
        return np.nan
    return float(np.interp(target, y[order], x[order]))


def main():
    for den, src, srcname in (('C4', VDB, 'VdB01'), ('C4', CSH, 'C&S03'),
                              ('O3', CSH, 'C&S03')):
        print(f'\n===== denominator {den}  (observed: {srcname}) =====')
        print(f'{"ion":>5} {"Zalpha":>7} {"obs ratio":>12} {"N/O":>8} {"N/H":>8}')
        for ion in ('N5', 'N4', 'N3'):
            r_obs, r_err = obs_ratio(src, ion, den)
            for mkey, path in MODEL_FILES.items():
                d = load(path)
                n = min(len(d[ion]), len(N_GRID))
                x = N_GRID[:n]
                y = (d[ion] / d[den])[:n]
                no = invert(x, y, r_obs)
                nh = no * Z_ALPHA[mkey]
                print(f'{ion:>5} {Z_ALPHA[mkey]:>7.0f} {r_obs:>12.4f} '
                      f'{no:>8.2f} {nh:>8.2f}')

    # geometric-mean combination across the two Z_alpha models, in N/H
    print('\n===== combined (geometric mean over Zalpha), N/H rel. solar =====')
    for den, src, srcname in (('C4', VDB, 'VdB01'), ('C4', CSH, 'C&S03'),
                              ('O3', CSH, 'C&S03')):
        print(f'  -- {den} / {srcname} --')
        for ion in ('N5', 'N4', 'N3'):
            r_obs, _ = obs_ratio(src, ion, den)
            vals = []
            for mkey, path in MODEL_FILES.items():
                d = load(path)
                n = min(len(d[ion]), len(N_GRID))
                v = invert(N_GRID[:n], (d[ion] / d[den])[:n], r_obs)
                if np.isfinite(v):
                    vals.append(v * Z_ALPHA[mkey])
            if vals:
                print(f'    {ion}: N/H = {np.exp(np.mean(np.log(vals))):.2f}'
                      f'   (span {min(vals):.2f} - {max(vals):.2f})')


if __name__ == '__main__':
    main()
