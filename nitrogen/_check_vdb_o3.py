#!/usr/bin/env python3
"""Quick check: nitrogen abundances from the VdB composite with O III] 1663.

VdB O III] 1663 = 0.480 +/- 0.028 (relative flux, Lya = 100), from their
Table 2.  Compare the N_x/O III] ratios and inferred (N/O)/(N/O)_sun between
the VdB and C&S03 composites, on the canonical Z_alpha = 3 LOC curves.
"""
from loc_nitrogen_abundance import VDB, CSH, obs_ratio
from plot_loc_nitrogen_composites import infer

for den in ('C4', 'O3'):
    print(f'--- denominator {den} ---')
    for ion in ('N5', 'N4', 'N3'):
        row = []
        for name, src in (('VdB01', VDB), ('C&S03', CSH)):
            r, re_ = obs_ratio(src, ion, den)
            x, lo, hi, off = infer(ion, src, den)
            flag = ' (extrap)' if off else ''
            row.append(f'{name}: ratio {r:6.3f}+/-{re_:5.3f}  '
                       f'X = {x:5.2f} [{lo:5.2f}, {hi:5.2f}]{flag}')
        print(f'  {ion}:  ' + '   |   '.join(row))
