"""Quick check: radial distribution of line luminosity in the velocity-resolved model."""
import numpy as np
from alpha.loc_gradient_fit import make_interpolators
from alpha.velocity_resolved_profiles import shell_luminosities, LOG_R_Q, M_BH_MSUN, G_CGS, MSUN_G, KMS

interps = make_interpolators()
logx, dL = shell_luminosities(interps, 0.0)
for key in ('C4', 'Si4O4' if 'Si4O4' in dL else 'Si4'):
    d = dL['C4'] if key == 'C4' else dL['Si4'] + dL['O4']
    d = d / d.sum()
    cum = np.cumsum(d)
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        i = np.searchsorted(cum, frac)
        r = 10**(LOG_R_Q + logx[i])
        sig = np.sqrt(G_CGS*M_BH_MSUN*MSUN_G/r)/KMS/np.sqrt(3)
        print(f"{key}: {int(frac*100)}% of L inside log x = {logx[i]:+.2f}  (sigma_los there = {sig:.0f} km/s)")
    print(f"{key}: fraction of L from x < 1: {cum[np.searchsorted(logx, 0.0)]:.3f}")
    print(f"{key}: dL/dlogx at logx = -1, 0, +1: "
          + ", ".join(f"{d[np.argmin(np.abs(logx - t))]:.4f}" for t in (-1, 0, 1)))
