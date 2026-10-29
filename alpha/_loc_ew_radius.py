#!/usr/bin/env python3
"""Emission-weighted CIV radius of the alpha-element LOC cloud distribution.

r_ref is the x=1 anchor; the clouds span x=r/r_Q in [0.1,10].  The
RM-comparable radius is the CIV-luminosity-weighted <x> times r_ref.
This reproduces the CIV LOC integrand of model_line_ratios() but tracks x.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, 'alpha')
import loc_gradient_fit as LG

LD = 2.59020684e15            # cm per light-day
INTERPS = LG.make_interpolators()


def ew_radius(log_r_ref, k, p, z_norm, dlq=0.0, n_x=400, n_n=24):
    """CIV emission-weighted <x> and <r> [ld] at logQ - LOGQ_REF = dlq."""
    logx = np.linspace(-1.0, 1.0, n_x)          # x = r/r_Q in [0.1, 10]
    logn = np.linspace(9.0, 12.0, n_n)
    logphi_ref = LG.LOGQ_REF - LG._LOG4PI - 2.0 * log_r_ref

    w_x = 10.0 ** (logx * (LG.GAMMA_LOC + 3.0))
    w_n = 10.0 ** (logn * (LG.BETA_N_LOC + 1.0))
    W = np.outer(w_n, w_x)

    logphi_Q = logphi_ref - 2.0 * logx + (1.0 - 2.0 * p) * dlq
    logZ = np.log10(z_norm) + k * logx + (p * k) * dlq
    logU = logphi_Q[None, :] - logn[:, None] - LG.LOG_C_LIGHT
    W_eff = W * (logphi_Q >= LG.LOGPHI_LO_LOC)[None, :] * LG._logu_weight(logU)

    logphi_c = np.clip(logphi_Q, LG.LOGPHI_GRID.min(), LG.LOGPHI_GRID.max())
    Z_c = np.clip(10.0 ** logZ, LG.Z_GRID.min(), LG.Z_GRID.max())
    LN, _ = np.meshgrid(logn, logx, indexing='ij')
    qpts = np.stack([LN,
                     np.broadcast_to(logphi_c, LN.shape),
                     np.broadcast_to(Z_c, LN.shape)], axis=-1).reshape(-1, 3)
    I_C4 = INTERPS['C4'](qpts).reshape(LN.shape)

    contrib = W_eff * I_C4                       # CIV luminosity per (n,x) cell
    cx = contrib.sum(axis=0)                     # collapse density -> per-x
    tot = cx.sum()
    x = 10.0 ** logx
    x_mean = np.sum(cx * x) / tot                # linear emission-weighted <x>
    logx_mean = np.sum(cx * logx) / tot          # log-space <log x>
    r_ref_ld = 10.0 ** log_r_ref / LD
    return x_mean, 10.0 ** logx_mean, x_mean * r_ref_ld, r_ref_ld


df = pd.read_csv('fits/alpha/loc_gradient_noA_Qref58/loc_fit_summary.csv')
med = dict(log_r_ref=df['log_r_ref'].median(), k=df['k'].median(),
           p=df['p'].median(), z_norm=df['Z_norm'].median())
print(f"median fit params: log_r_ref={med['log_r_ref']:.2f}, k={med['k']:.2f}, "
      f"p={med['p']:.2f}, z_norm={med['z_norm']:.2f}\n")

print(f'{"case":<26}{"<x>lin":>9}{"<x>log":>9}{"r_ref[ld]":>11}'
      f'{"<r>_CIV[ld]":>13}')
print('-' * 68)
for label, lrr in [('median fit', med['log_r_ref']),
                    ('prior floor (17.0)', 17.0),
                    ('16th pct (17.06)', 17.06),
                    ('84th pct (17.50)', 17.50)]:
    xl, xlg, rew, rref = ew_radius(lrr, med['k'], med['p'], med['z_norm'])
    print(f'{label:<26}{xl:9.2f}{xlg:9.2f}{rref:11.1f}{rew:13.1f}')

# also at a typical object luminosity (Q ~ 1e55, i.e. dlq = -3)
xl, xlg, rew, rref = ew_radius(med['log_r_ref'], med['k'], med['p'],
                               med['z_norm'], dlq=-3.0)
print(f'\nmedian fit, evaluated at logQ=55 (dlq=-3): <x>lin={xl:.2f}, '
      f'<r>_CIV = {xl*rref*10**(med["p"]*-3.0):.1f} ld '
      f'(r_Q breathed in by 10^(p*dlq))')
