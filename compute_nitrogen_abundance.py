"""
Compute nitrogen abundance (N/H)/(N/H)_sun as a function of BLR radius
from Cloudy singlezone nitrogen models, and save to a .dat file.

Based on Cloudy_LOC_metal_series.ipynb cells 12-16.

Input:
    - Cloudy model files in singlezone_nitrogen_series/
    - Observed line ratios from Van den Berk et al. (2001)

Output:
    - nitrogen_abundance_vs_r.dat (r_blr [ld], Z_best, Z_low, Z_high for each model)

Usage:
    python compute_nitrogen_abundance.py
"""

import numpy as np
import pandas as pd
import os
from scipy.interpolate import interp1d

# ============================================================================
# Observed line fluxes (Van den Berk et al. 2001)
# ============================================================================
NV1240_obs = 2.461;   NV1240_err = 0.189
NIV1718_obs = 0.258;  NIV1718_err = 0.027
NIII1750_obs = 0.382; NIII1750_err = 0.021
CIV1549_obs = 25.29;  CIV1549_err = 0.11

R_N3C4 = NIII1750_obs / CIV1549_obs
R_N4C4 = NIV1718_obs / CIV1549_obs
R_N5C4 = NV1240_obs / CIV1549_obs

R_N3C4_err = R_N3C4 * np.sqrt((NIII1750_err / NIII1750_obs)**2 + (CIV1549_err / CIV1549_obs)**2)
R_N4C4_err = R_N4C4 * np.sqrt((NIV1718_err / NIV1718_obs)**2 + (CIV1549_err / CIV1549_obs)**2)
R_N5C4_err = R_N5C4 * np.sqrt((NV1240_err / NV1240_obs)**2 + (CIV1549_err / CIV1549_obs)**2)

R_obs = {'N3': R_N3C4, 'N4': R_N4C4, 'N5': R_N5C4}
R_err = {'N3': R_N3C4_err, 'N4': R_N4C4_err, 'N5': R_N5C4_err}

# ============================================================================
# Load Cloudy nitrogen models (singlezone, 3 metallicities)
# ============================================================================
Z_grid_nitro = np.arange(0.1, 10.1, 0.2)
basepath_nitro = '/Users/jiamuh/c23.01/my_models/singlezone_nitrogen_series'

line_names_nitro = {
    'N5': 'blnd 1240.00A',
    'N4': 'blnd 1486.00A',
    'N3': 'blnd 1750.00A',
    'C4': 'blnd 1549.00A',
}

intensity = {}
for mkey, suffix in [('m1', 'm1'), ('m3', 'm3'), ('m10', 'm10')]:
    fname = os.path.join(basepath_nitro,
                         f'strong_n10_phi19_N23_{suffix}_v0_LineList_BLR_Fe2.txt')
    df = pd.read_csv(fname, sep='\t', header=0, comment='#')
    intensity[mkey] = {key: df[col].values for key, col in line_names_nitro.items()}

# ============================================================================
# Find best-fit nitrogen abundance for each species
# ============================================================================
def find_best_Z_interp(R_mod, R_obs, R_err, Z_grid):
    """Interpolate model ratio vs Z to find Z that matches observed ratio."""
    Z_best, Z_low, Z_high = {}, {}, {}
    for key in ['N3', 'N4', 'N5']:
        ratio = R_mod[key]
        obs, err = R_obs[key], R_err[key]

        mask = np.isfinite(ratio)
        Z, R = Z_grid[mask], ratio[mask]
        order = np.argsort(Z)
        Z, R = Z[order], R[order]

        f = interp1d(R, Z, bounds_error=False, fill_value='extrapolate')

        Z_best[key] = float(f(obs))
        Z_low[key] = float(f(obs - err))
        Z_high[key] = float(f(obs + err))

    return Z_best, Z_low, Z_high


results = {}
for mkey in ['m1', 'm3', 'm10']:
    R_mod = {k: intensity[mkey][k] / intensity[mkey]['C4'] for k in ['N3', 'N4', 'N5']}
    Zb, Zl, Zh = find_best_Z_interp(R_mod, R_obs, R_err, Z_grid_nitro)
    results[mkey] = {'Z_best': Zb, 'Z_low': Zl, 'Z_high': Zh}

# ============================================================================
# Compute BLR radii (anchored to NV at 2 light-days)
# ============================================================================
phi_best = {'N3': 17, 'N4': 19, 'N5': 21}
phi_range = {'N3': 0.5, 'N4': 1.5, 'N5': 1.5}
phi_ref = phi_best['N5']  # 21
r_ref = 2.0  # light-days

r_blr = {}
r_range_blr = {}
for line in ['N3', 'N4', 'N5']:
    r_blr[line] = r_ref * 10**(-0.5 * (phi_best[line] - phi_ref))
    dphi = phi_range[line]
    r_low = r_ref * 10**(-0.5 * ((phi_best[line] + dphi) - phi_ref))
    r_high = r_ref * 10**(-0.5 * ((phi_best[line] - dphi) - phi_ref))
    r_range_blr[line] = (r_low, r_high)

# ============================================================================
# Save results
# ============================================================================
outfile = 'nitrogen_abundance_vs_r.dat'
with open(outfile, 'w') as f:
    f.write("# Nitrogen abundance (N/H)/(N/H)_sun vs BLR radius\n")
    f.write("# From Cloudy singlezone models + Van den Berk et al. (2001) observations\n")
    f.write("# Columns: species, r_blr [ld], r_low [ld], r_high [ld], "
            "model, Z_best, Z_low, Z_high\n")
    for line in ['N3', 'N4', 'N5']:
        for mkey in ['m1', 'm3', 'm10']:
            r = r_blr[line]
            r_lo, r_hi = r_range_blr[line]
            Zb = results[mkey]['Z_best'][line]
            Zl = results[mkey]['Z_low'][line]
            Zh = results[mkey]['Z_high'][line]
            f.write(f"{line}  {r:.6f}  {r_lo:.6f}  {r_hi:.6f}  "
                    f"{mkey}  {Zb:.6f}  {Zl:.6f}  {Zh:.6f}\n")

print(f"Saved {outfile}")
print(f"\nBLR radii (light-days):")
for line in ['N3', 'N4', 'N5']:
    r_lo, r_hi = r_range_blr[line]
    print(f"  {line}: r = {r_blr[line]:.2f} ld  [{r_lo:.2f}, {r_hi:.2f}]")

print(f"\nNitrogen abundances (N/H)/(N/H)_sun:")
for mkey in ['m1', 'm3', 'm10']:
    print(f"  {mkey}:")
    for line in ['N3', 'N4', 'N5']:
        Zb = results[mkey]['Z_best'][line]
        Zl = results[mkey]['Z_low'][line]
        Zh = results[mkey]['Z_high'][line]
        print(f"    {line}: {Zb:.3f} [{Zl:.3f}, {Zh:.3f}]")
