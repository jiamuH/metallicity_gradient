"""Quick: LOC-averaged MgII/CIV when clouds above a per-cloud ratio cap are
removed.  Does cutting 'unphysical' high-ratio clouds fix the average?  Scratch.
"""
import numpy as np
from alpha.loc_gradient_fit import (make_interpolators, LOGQ_REF, _LOG4PI,
                                    LOGPHI_GRID, GAMMA_LOC, BETA_N_LOC,
                                    LOGPHI_LO_LOC, LOG_C_LIGHT, MGII_SCALE, _logu_weight)

K, P, Z, LOGQ, LRR = 0.0, 0.35, 3.0, 55.0, 17.5
interps = make_interpolators()
n_x, n_n = 80, 24
logx = np.linspace(-1, 1, n_x)
logn = np.linspace(9, 12, n_n)
logphi_Q = (LOGQ_REF - _LOG4PI - 2 * LRR) - 2 * logx + (1 - 2 * P) * (LOGQ - LOGQ_REF)
W = np.outer(10.0 ** (logn * (BETA_N_LOC + 1)), 10.0 ** (logx * (GAMMA_LOC + 3)))
LN, LPH = np.meshgrid(logn, logphi_Q, indexing='ij')
logU = LPH - LN - LOG_C_LIGHT
W_eff = W * (logphi_Q >= LOGPHI_LO_LOC)[None, :] * _logu_weight(logU)
logphi_c = np.clip(logphi_Q, LOGPHI_GRID.min(), LOGPHI_GRID.max())
qp = np.stack([LN, np.broadcast_to(logphi_c, LN.shape), np.full(LN.shape, Z)], -1).reshape(-1, 3)
eMg = interps['Mg2'](qp).reshape(LN.shape)
eC4 = interps['C4'](qp).reshape(LN.shape)
rho = MGII_SCALE * eMg / eC4
LMg0 = np.sum(W_eff * eMg)

print(f"point: k={K} p={P} Z={Z} logQ={LOGQ} r_ref={LRR};  observed log(MgII/CIV) = -0.389")
print(f"{'cap rho<=':>10} {'LOC log(MgII/CIV)':>18} {'L_Mg removed':>14}")
for T in [np.inf, 100, 30, 10, 3, 1]:
    m = rho <= T
    LMg, LC4 = np.sum(W_eff * eMg * m), np.sum(W_eff * eC4 * m)
    avg = np.log10(MGII_SCALE * LMg / LC4)
    print(f"{T:>10g} {avg:>18.3f} {1 - LMg / LMg0:>13.1%}")
