"""
Extract the two photoionization sensitivities that the breathing mock needs,
from the Cloudy LOC grid, using the existing LOC integral (model_line_ratios):

  S_U = d ln(ratio) / d ln U |_Z      (ionization sensitivity, fixed metallicity)
  A_Z = d ln(ratio) / d ln Z |_U      (abundance sensitivity, fixed ionization)

for ratio = NV/CIV and (SiIV+OIV)/CIV.

How each is isolated within the LOC model:
  - Set gradient k = 0 so metallicity is uniform (= z_norm) and does not move
    with luminosity.
  - In the model, a cloud at fixed (x, n) sees  log phi = ... + (1-2p) dlogQ,
    so  d log U / d log Q = (1 - 2p)  at fixed cloud. Thus varying logQ rigidly
    shifts the whole ionization comb, and
        S_U = [d log10(ratio)/d log10 Q]_{k=0} / (1 - 2p).
    Using p = 0 makes the divisor 1 (cleanest); checked against p = 0.4.
  - A_Z: vary z_norm at fixed logQ, k = 0:  A_Z = d log10(ratio)/d log10(z_norm).

Operating point: log_r_ref = LOG_R_REF_PLOT (17.7), z_norm = 3, logQ ~ 55.5,
which puts the cloud distribution at log phi ~ 19, log U ~ -2 (canonical BLR).

Run:
    ~/miniconda3/envs/pypeit/bin/python alpha/_extract_SU_AZ.py
"""

import numpy as np
from loc_gradient_fit import (load_cloudy_grid, make_interpolators,
                              model_line_ratios, LOG_R_REF_PLOT,
                              LOGQ_REF, _LOG4PI, LOG_C_LIGHT)

RATIOS = ['NV/CIV', '(SiIV+OIV)/CIV']
INTERPS = make_interpolators(load_cloudy_grid())
LOGN_MID = 10.5   # centre of the model's default cloud density range [9, 12]


def logU_center(logQ, log_r_ref=LOG_R_REF_PLOT, p=0.0):
    """Nominal central ionization log U of the cloud distribution at this logQ:
    flux at x=1 minus the mid density and c.  (p=0 => log phi shifts 1:1 with logQ.)"""
    logphi_ref = LOGQ_REF - _LOG4PI - 2.0 * log_r_ref
    logphi_x1 = logphi_ref + (1.0 - 2.0 * p) * (logQ - LOGQ_REF)
    return logphi_x1 - LOGN_MID - LOG_C_LIGHT


def logratio(logQ, z_norm, k, p, log_r_ref=LOG_R_REF_PLOT):
    """log10 of each LOC line ratio at one operating point."""
    r, _, _ = model_line_ratios(INTERPS, logQ, k=k, p=p,
                                log_r_ref=log_r_ref, z_norm=z_norm)
    return {key: np.log10(r[key][0]) for key in RATIOS}


def deriv(f_plus, f_minus, h):
    return {key: (f_plus[key] - f_minus[key]) / (2.0 * h) for key in RATIOS}


def S_U_at(logQ, z_norm, p=0.0, h=0.1):
    """d log10(ratio)/d log10 U  (= /(1-2p) of the logQ derivative at k=0)."""
    fp = logratio(logQ + h, z_norm, k=0.0, p=p)
    fm = logratio(logQ - h, z_norm, k=0.0, p=p)
    d = deriv(fp, fm, h)
    return {key: d[key] / (1.0 - 2.0 * p) for key in RATIOS}


def A_Z_at(logQ, z_norm, h=0.04):
    """d log10(ratio)/d log10(z_norm) at fixed logQ, k=0."""
    lz = np.log10(z_norm)
    fp = logratio(logQ, 10.0 ** (lz + h), k=0.0, p=0.0)
    fm = logratio(logQ, 10.0 ** (lz - h), k=0.0, p=0.0)
    return deriv(fp, fm, h)


if __name__ == '__main__':
    logQ0, z0 = 55.5, 3.0
    print(f"Reference: logQ={logQ0}, z_norm={z0} Zsun, log_r_ref={LOG_R_REF_PLOT}, "
          f"log U = {logU_center(logQ0):.2f}\n")

    su = S_U_at(logQ0, z0, p=0.0)
    su_chk = S_U_at(logQ0, z0, p=0.4)      # p-independence check
    az = A_Z_at(logQ0, z0)

    print(f"{'ratio':>16} | {'S_U':>7} {'(p=0.4 chk)':>11} | {'A_Z':>7} | {'S_U/A_Z':>8}")
    print("-" * 62)
    for key in RATIOS:
        print(f"{key:>16} | {su[key]:7.3f} {su_chk[key]:11.3f} | "
              f"{az[key]:7.3f} | {su[key]/az[key]:8.3f}")

    # --- S_U and A_Z extracted across a range of ionization, up to high log U ---
    # ionization set by logQ (p=0): each step of +0.5 in logQ raises log U by +0.5.
    logQ_list = [55.0, 55.5, 56.0, 56.5, 57.0]
    print("\nS_U and A_Z vs ionization (z_norm=3, both ratios):")
    print(f"   {'log U':>6} | {'S_U(NV)':>8} {'A_Z(NV)':>8} {'S_U/A_Z':>8} | "
          f"{'S_U(SiO)':>9} {'A_Z(SiO)':>9} {'S_U/A_Z':>8}")
    print("   " + "-" * 74)
    for lq in logQ_list:
        lU = logU_center(lq)
        s = S_U_at(lq, z0, p=0.0)
        a = A_Z_at(lq, z0)
        rN = s['NV/CIV'] / a['NV/CIV']
        rS = s['(SiIV+OIV)/CIV'] / a['(SiIV+OIV)/CIV']
        print(f"   {lU:6.2f} | {s['NV/CIV']:8.3f} {a['NV/CIV']:8.3f} {rN:8.3f} | "
              f"{s['(SiIV+OIV)/CIV']:9.3f} {a['(SiIV+OIV)/CIV']:9.3f} {rS:8.3f}")

    # absolute log10(line ratio) at the two reference ionizations (for the mock)
    print("\nAbsolute log10(line ratio) at z_norm=3:")
    for lq in [55.5, 56.5]:
        lr = logratio(lq, z0, k=0.0, p=0.0)
        print(f"   log U={logU_center(lq):+.2f}: "
              f"log(NV/CIV)={lr['NV/CIV']:+.3f}, "
              f"log((SiIV+OIV)/CIV)={lr['(SiIV+OIV)/CIV']:+.3f}")

    # mock inputs (per-object tilts) at low vs high ionization, NV/CIV
    a_breath = 0.25
    print(f"\nMock per-object tilts (NV/CIV, breathing a={a_breath}):")
    for lq in [55.5, 56.5]:
        lU = logU_center(lq)
        s = S_U_at(lq, z0, p=0.0)['NV/CIV']
        az_h = A_Z_at(lq, z0)['NV/CIV']
        cZ = (s / az_h) * (1.0 - 2.0 * a_breath)
        print(f"   log U={lU:+.2f}: S_U={s:.3f} A_Z={az_h:.3f} -> "
              f"s_b(M)={cZ/(0.5-a_breath):+.2f}, s_b(lambda)={cZ/(0.5+a_breath):+.2f}")
