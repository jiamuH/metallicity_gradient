"""Diagnostic: why does the model MgII/CIV run large in the heatmap?

Replicates the LOC integral of `model_line_ratios` but reports the per-line
LOC-integrated luminosities (Mg2, C4) and the emission-weighted ionizing flux
of each line, scanned over log r_ref (the heatmap c-row variable).

Run from repo root:
    python3 -m alpha.diag_mgii_civ
"""
import numpy as np
from alpha.loc_gradient_fit import (make_interpolators, LOGQ_REF, _LOG4PI,
                                    LOGPHI_GRID, Z_GRID, GAMMA_LOC, BETA_N_LOC,
                                    LOGPHI_LO_LOC, LOG_R_REF_PLOT, MGII_SCALE)

K, P, ZN, LOGQ = 0.0, 0.35, 3.0, 55.0
LINES = ('Mg2', 'C4')


def loc_diag(interps, log_r_ref, logQ=LOGQ, k=K, p=P, z_norm=ZN,
             logx_lo=-1.0, logx_hi=1.0, n_x=80, logn_lo=9.0, logn_hi=12.0, n_n=24):
    """One (k, p, log_r_ref, logQ) point: per-line LOC luminosity and the
    emission-weighted log phi.  Matches model_line_ratios exactly."""
    logphi_ref = LOGQ_REF - _LOG4PI - 2.0 * log_r_ref
    logx = np.linspace(logx_lo, logx_hi, n_x)
    logn = np.linspace(logn_lo, logn_hi, n_n)
    w_x = 10.0 ** (logx * (GAMMA_LOC + 3.0))
    w_n = 10.0 ** (logn * (BETA_N_LOC + 1.0))
    W = np.outer(w_n, w_x)
    LN, _ = np.meshgrid(logn, logx, indexing='ij')
    dlq = logQ - LOGQ_REF
    logphi_Q = logphi_ref - 2.0 * logx + (1.0 - 2.0 * p) * dlq
    logZ = np.log10(z_norm) + k * logx + (p * k) * dlq
    valid_x = logphi_Q >= LOGPHI_LO_LOC
    W_eff = W * valid_x[None, :]
    logphi_c = np.clip(logphi_Q, LOGPHI_GRID.min(), LOGPHI_GRID.max())
    Z_c = np.clip(10.0 ** logZ, Z_GRID.min(), Z_GRID.max())
    n_clip = int(np.sum(logphi_Q != logphi_c))
    LPQ = np.broadcast_to(logphi_c, LN.shape)
    ZZ = np.broadcast_to(Z_c, LN.shape)
    qpts = np.stack([LN, LPQ, ZZ], axis=-1).reshape(-1, 3)
    out = {}
    for key in LINES:
        contrib = W_eff * interps[key](qpts).reshape(LN.shape)        # (n_n, n_x)
        per_x = contrib.sum(axis=0)
        L = float(per_x.sum())
        phi_ew = float(np.sum(logphi_c * per_x) / per_x.sum()) if per_x.sum() > 0 else np.nan
        out[key] = (L, phi_ew)
    return out, logphi_ref, (logphi_Q.min(), logphi_Q.max()), n_clip


def main():
    interps = make_interpolators()
    print(f"k={K}, p={P}, Z_norm={ZN} Zsun, logQ={LOGQ};  MGII_SCALE={MGII_SCALE}")
    print(f"GAMMA_LOC={GAMMA_LOC}, BETA_N_LOC={BETA_N_LOC};  "
          f"Cloudy phi grid = [{LOGPHI_GRID.min():.0f}, {LOGPHI_GRID.max():.0f}]\n")
    hdr = (f"{'log r_ref':>9} {'logphi_ref':>10} {'cloud phi span':>17} "
           f"{'log(MgII/CIV)':>14} {'phiEW(Mg)':>10} {'phiEW(C4)':>10} {'clip':>5}")
    print(hdr)
    print('-' * len(hdr))
    for lr in [16.30, 16.50, 16.70, 16.95, 17.20, 17.45, 17.70, 17.95]:
        d, lpr, (pmn, pmx), nclip = loc_diag(interps, lr)
        Lmg, phimg = d['Mg2']
        Lc4, phic4 = d['C4']
        logratio = np.log10(MGII_SCALE * Lmg / Lc4)
        span = f"[{pmn:.1f}, {pmx:.1f}]"
        print(f"{lr:9.2f} {lpr:10.2f} {span:>17} "
              f"{logratio:14.3f} {phimg:10.2f} {phic4:10.2f} {nclip:5d}")
    print(f"\nobserved MgII/CIV (table, bounding box): log = -0.36  (range -0.60..-0.12)")
    print(f"heatmap a/b panels: fixed log r_ref = {LOG_R_REF_PLOT}")
    print(f"heatmap c-row     : scans log r_ref in [16.95, 17.95]")


if __name__ == '__main__':
    main()
