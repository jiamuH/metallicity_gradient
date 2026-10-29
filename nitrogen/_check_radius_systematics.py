"""Radius-centroid systematics for the nitrogen joint fit (SIG_R, RHO_R_OUTER in
joint_gradient_fit.py): vary the LOC weights Gamma, beta over [-1.5, -0.5] and
report the spread of each ion's emission-weighted radius (N V anchored at 2 ld),
and the correlation between the N IV] and N III] centroid shifts.  Done for the
metals-1 grid (the old radii) and the metals-3 grid (the Z_alpha = 3 radii).

Run (from the project root): python3 nitrogen/_check_radius_systematics.py
"""
import itertools

import numpy as np

from loc_nitrogen_abundance import consistent_loc_radii, radii_chunks

M1 = [('strong_LOC_m1_N23_LineList_BLR_Fe2_flux.txt', np.arange(17.0, 23.5, 0.5), np.arange(9.0, 12.5, 0.5))]
VALUES = (-1.5, -1.0, -0.5)


def main():
    for tag, chunks in (('metals 1', M1), ('metals 3', radii_chunks('N23_m3'))):
        base = consistent_loc_radii(chunks=chunks)
        shifts = {i: [] for i in ('N5', 'N4', 'N3')}
        for g, b in itertools.product(VALUES, VALUES):
            r = consistent_loc_radii(beta=b, gamma=g, chunks=chunks)
            for i in shifts:
                shifts[i].append(np.log10(r[i][0] / base[i][0]))
        print(f'--- {tag}: baseline radii ' + ', '.join(f'{i} {base[i][0]:.2f} ld' for i in shifts))
        for i, s in shifts.items():
            s = np.array(s)
            print(f'  {i}: shift range {s.min():+.2f} to {s.max():+.2f} dex, rms {np.sqrt(np.mean(s ** 2)):.2f} dex')
        rho = np.corrcoef(shifts['N4'], shifts['N3'])[0, 1]
        print(f'  N IV] - N III] shift correlation: {rho:.2f}')
        ratio = np.array(shifts['N3']) - np.array(shifts['N4'])
        print(f'  log(r_N3/r_N4) shift range {ratio.min():+.2f} to {ratio.max():+.2f} dex')


if __name__ == '__main__':
    main()
