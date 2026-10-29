#!/usr/bin/env python3
"""Effect of the abundance-side grid selection on the LOC emission radii.

The published radii (loc_emission_weighted_r.py) integrate the line emission
over the FULL stitched grid (log phi 16-23, log n 7-12) with no matter-bounded
cut.  The abundance inference uses log phi 17-21, log n 9-12, and removes
matter-bounded clouds (log phi - log n < log(alpha_B N_H) = 10.41 at N23).
This script recomputes (r_ew, r25, r75) under both selections, with the
density average taken over the SURVIVING cells (as in loc_nitrogen_abundance),
and re-anchors N V to 2 light-days in each case.

Diagnostic only; writes nothing.
"""
import numpy as np

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz

import loc_emission_weighted_r as L

LD_PER_CM = L.LD_PER_CM
PHI, LOGN = L.PHI_AX_FULL, L.LOGN_AX_FULL


def radii(mask2d, beta=-1.0, gamma=-1.0, r_nv_ld=2.0):
    em = L._load_line_intensity()
    n_lin = 10 ** LOGN
    w_n = n_lin ** (beta + 1)

    em_loc = {}
    for ln in L.LINE_INFO:
        prof = np.zeros(len(PHI))
        for ip in range(len(PHI)):
            keep = mask2d[ip]
            if keep.any():
                prof[ip] = np.sum(em[ln][ip, keep] * w_n[keep]) / np.sum(w_n[keep])
        em_loc[ln] = prof

    logphi_dense = np.linspace(PHI.min(), PHI.max(), 400)
    logr_dense = (54.0 - np.log10(4 * np.pi) - logphi_dense) / 2.0
    r_dense = 10 ** logr_dense
    raw = {}
    for ln in L.LINE_INFO:
        em_d = np.interp(logphi_dense, PHI, em_loc[ln])
        f = np.maximum(em_d, 0.0) * r_dense ** (gamma + 2)
        order = np.argsort(r_dense)
        r_s, f_s = r_dense[order], f[order]
        L_tot = np.trapezoid(f_s, r_s)
        r_ew = 10 ** (np.trapezoid(np.log10(r_s) * f_s, r_s) / L_tot)
        cum = np.concatenate([[0.0],
                              np.cumsum(0.5 * (f_s[:-1] + f_s[1:]) * np.diff(r_s))])
        cum /= cum[-1]
        raw[ln] = (r_ew, np.interp(0.25, cum, r_s), np.interp(0.75, cum, r_s))
    scale = (r_nv_ld / LD_PER_CM) / raw['N5'][0]
    return {ln: tuple(v * scale * LD_PER_CM for v in t) for ln, t in raw.items()}


def main():
    pp, nn = np.meshgrid(PHI, LOGN, indexing='ij')
    full = np.ones_like(pp, bool)
    consistent = (nn >= 9.0) & (pp >= 17.0) & (pp <= 21.0) & ((pp - nn) < 10.41)

    for tag, mask in (('full grid (published)', full),
                      ('abundance-consistent', consistent)):
        r = radii(mask)
        print(f'\n{tag}:  ({int(mask.sum())} of {mask.size} cells)')
        for ln in ('N5', 'N4', 'N3'):
            rew, lo, hi = r[ln]
            print(f'  {ln}: r_ew = {rew:6.2f} ld   25-75%: [{lo:6.2f}, {hi:6.2f}]')


if __name__ == '__main__':
    main()
