"""How strongly do the LOC-integrated line ratios respond to metallicity at both
ends of the Cloudy grid?  Adds the low-Z (0.01, 0.02, 0.04 Z_sun) and high-Z
(32, 56, 100 Z_sun) extension runs to the standard 0.05-20 Z_sun grid and
evaluates, at k = 0 (uniform metallicity across the LOC window),

    MgII/CIV and (SiIV+OIV)/CIV   and   d log(ratio) / d log Z

at every metallicity node.  The LOC geometry uses the median fitted p and the
median log Q of the sample, with log r_ref at the 16th, 50th and 84th
percentiles of the fits.  An extension file that does not exist yet (grid
still running) is skipped.

Run: python3 -m alpha._check_z_saturation_extended
"""
import os

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import alpha.loc_gradient_fit as lgf
from alpha.plot_Z_profiles import _logQ_per_object

CLOUDY_DIR = '/Users/jiamuh/c23.01/my_models/loc_metal'
EXT_FILES = {   # label: (line-list file, metallicities in Z_sun)
    'low':  (f'{CLOUDY_DIR}/strong_LOC_varym_lowZ_N25_LineList_BLR_Fe2.txt',  10.0 ** np.array([-2.0, -1.7, -1.4])),
    'high': (f'{CLOUDY_DIR}/strong_LOC_varym_highZ_N25_LineList_BLR_Fe2.txt', 10.0 ** np.array([1.5, 1.75, 2.0])),
}
FIT_CSV = 'fits/alpha/loc_gradient_noA_Qref58/loc_fit_summary.csv'
LINES4 = ('Mg2', 'C4', 'Si4', 'O4')
lgf.LINE_COLS.setdefault('He2', 'he 2 1640.41A')     # also read He II 1640 from the line lists


def extended_interpolator(lines=LINES4):
    """Interpolator over (log n_H, log phi, Z) on the standard grid plus whatever
    extension runs exist, with a trailing axis holding `lines` (four keys of
    lgf.LINE_COLS, in the order model_ratios_at expects).  Returns (interpolator, Z nodes)."""
    blocks = [(lgf.Z_GRID, lgf.load_cloudy_grid())]
    for label, (path, zs) in EXT_FILES.items():
        try:
            blocks.append((zs, lgf._load_one_cloudy_file(path, len(lgf.LOGPHI_GRID), len(zs))))
        except (FileNotFoundError, ValueError) as err:     # missing, or still being written
            print(f'  {label}-Z extension skipped ({type(err).__name__}): {os.path.basename(path)}')
    blocks.sort(key=lambda b: b[0][0])
    z = np.concatenate([b[0] for b in blocks])
    stack = np.stack([np.concatenate([b[1][key] for b in blocks], axis=2) for key in lines], axis=-1)
    interp = RegularGridInterpolator((lgf.LOGN_GRID, lgf.LOGPHI_GRID, z), np.maximum(stack, 1e-30),
                                     bounds_error=False, fill_value=None)
    return interp, z


def main():
    interp, z = extended_interpolator()
    lgf.Z_GRID = z      # model_ratios_at clips Z to lgf.Z_GRID; widen it to the extended range
    fits = pd.read_csv(FIT_CSV).dropna(subset=['k'])
    logq = np.median(_logQ_per_object(fits.rm_id.tolist()).values)
    p = fits.p.median()
    lrr = np.percentile(fits.log_r_ref, [16, 50, 84])
    print(f'k = 0, p = {p:.2f}, log Q = {logq:.2f}; metallicity nodes {z.min():.2f}-{z.max():.0f} Z_sun\n')

    lz = np.log10(z)
    for lr, tag in zip(lrr, ('16th', '50th', '84th')):
        mg, si = np.array([model_at(interp, logq, p, lr, zz) for zz in z]).T
        dmg = np.gradient(np.log10(mg), lz)
        dsi = np.gradient(np.log10(si), lz)
        print(f'--- log r_ref = {lr:.2f} ({tag} percentile of the fits)')
        print(f'{"Z [Z_sun]":>10s} {"MgII/CIV":>9s} {"dlog/dlogZ":>11s} {"(SiIV+OIV)/CIV":>15s} {"dlog/dlogZ":>11s}')
        for i in range(len(z)):
            if z[i] < 0.1 or z[i] >= 20 or z[i] in (0.25, 0.55, 1.0, 3.0, 10.0, 15.0):
                print(f'{z[i]:10.3g} {mg[i]:9.3f} {dmg[i]:+11.2f} {si[i]:15.3f} {dsi[i]:+11.2f}')
        print()


def model_at(interp, logq, p, log_r_ref, z_norm):
    mg, si = lgf.model_ratios_at(interp, logq, 0.0, p, log_r_ref, z_norm)
    return mg[0], si[0]


if __name__ == '__main__':
    main()
