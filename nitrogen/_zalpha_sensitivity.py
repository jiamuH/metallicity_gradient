#!/usr/bin/env python3
"""How much does the inferred N/O depend on the assumed Z_alpha?

Each nitrogen line is inverted at its own radius, with Z_alpha taken from the
alpha-element analysis (Sec. 3).  This tabulates the inferred (N/O)/(N/O)_sun
for Z_alpha = 1, 3, 10 Zsun, i.e. the systematic carried by that assumption,
using the LOC models (separable X and Z dependence, as in solve_N_and_Z.py).

Diagnostic only; writes nothing.
"""
import numpy as np

import solve_N_and_Z as S
from loc_nitrogen_abundance import load_grid, obs_ratio, invert, VDB, CSH

RADII = {'N5': 2.00, 'N4': 4.83, 'N3': 8.63}     # light-days
Z_LIST = (1.0, 3.0, 10.0)


def main():
    S.GRID_X = load_grid('N25_m1')
    S.GRID_Z = S.load_varym()
    gx = np.logspace(np.log10(0.05), np.log10(9.0), 400)

    for label, src, den in (('VdB01 / C IV', VDB, 'C4'),
                            ('C&S03 / O III]', CSH, 'O3')):
        print(f'\n=== {label} ===')
        print(f'  {"ion":>5} {"r[ld]":>7} ' +
              ' '.join(f'{"Z=" + str(int(z)):>9}' for z in Z_LIST) +
              f' {"spread":>8}')
        for ion in ('N5', 'N4', 'N3'):
            r_obs, _ = obs_ratio(src, ion, den)
            vals = []
            for z in Z_LIST:
                model, _ = S.build_model(ion, den, gx, np.full_like(gx, z))
                x, off = invert(gx, model, r_obs)
                vals.append((x, off))
            cells = ' '.join(f'{v:9.2f}' + ('*' if o else ' ')
                             for v, o in vals)
            span = max(v for v, _ in vals) / min(v for v, _ in vals)
            print(f'  {ion:>5} {RADII[ion]:>7.2f} {cells} {span:>7.1f}x')
        print('  (* = extrapolated beyond the model grid)')


if __name__ == '__main__':
    main()
