"""Why is the LOC-integrated C IV/He II non-monotonic in Z?  Split it into
C IV/H-beta and He II/H-beta (H-beta tracks the absorbed hydrogen-ionizing
photons) at k = 0, median p, median log Q and median log r_ref, 0.01-100 Z_sun.

Run: python3 -m alpha._check_c4_he2_turnover
"""
import numpy as np
import pandas as pd

import alpha.loc_gradient_fit as lgf
from alpha._check_z_saturation_extended import extended_interpolator, model_at, FIT_CSV
from alpha.plot_Z_profiles import _logQ_per_object

lgf.LINE_COLS.setdefault('Hb', 'H  1 4861.32A')


def main():
    fits = pd.read_csv(FIT_CSV).dropna(subset=['k'])
    logq = np.median(_logQ_per_object(fits.rm_id.tolist()).values)
    p, lrr = fits.p.median(), fits.log_r_ref.median()
    out = {}
    for name, lines in (('C IV/Hb', ('C4', 'Hb', 'C4', 'C4')), ('He II/Hb', ('He2', 'Hb', 'C4', 'C4'))):
        lgf.Z_GRID = np.concatenate([lgf._Z_SUBSOLAR, lgf._Z_SUPER])
        interp, z = extended_interpolator(lines)
        lgf.Z_GRID = z
        out[name] = np.array([model_at(interp, logq, p, lrr, zz)[0] for zz in z])
    i_sun = np.argmin(abs(z - 1))
    print(f'{"Z [Z_sun]":>10s} {"log C IV/Hb":>12s} {"log He II/Hb":>12s} {"log C IV/He II":>15s}   (each relative to Z_sun)')
    for i in range(len(z)):
        if z[i] < 0.06 or z[i] > 19 or z[i] in (0.15, 0.25, 0.35, 0.45, 0.65, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0):
            c = np.log10(out['C IV/Hb'][i] / out['C IV/Hb'][i_sun])
            h = np.log10(out['He II/Hb'][i] / out['He II/Hb'][i_sun])
            print(f'{z[i]:10.3g} {c:+12.2f} {h:+12.2f} {c - h:+15.2f}')


if __name__ == '__main__':
    main()
