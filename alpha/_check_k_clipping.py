"""Where do the k-railed SDSS-RM fits sit relative to the Cloudy metallicity
grid (0.05-20 Z_sun)?  For each object, evaluate the model metallicity across
the LOC window at the object's median log Q, exactly as in model_ratios_at:

    log Z(x) = log Z_ref + k log x + p k (log Q - log Q_ref),   log x in [-1, 1],

using the fitted p and Z_ref, at k = +-1 and +-2.  Report the window range and
the fraction of the window (in log x) that falls outside the grid.

Run: python3 -m alpha._check_k_clipping
"""
import numpy as np
import pandas as pd

from alpha.loc_gradient_fit import Z_GRID, LOGQ_REF, _LOC_LOGX_HALFWIDTH
from alpha.plot_Z_profiles import _logQ_per_object

FIT_CSV = 'fits/alpha/loc_gradient_noA_Qref58/loc_fit_summary.csv'
IDS = 'fits/alpha/k_railed_ids.txt'
LZ_LO, LZ_HI = np.log10(Z_GRID.min()), np.log10(Z_GRID.max())


def window(lz_ref, p, k, logq, n=201):
    logx = np.linspace(-_LOC_LOGX_HALFWIDTH, _LOC_LOGX_HALFWIDTH, n)
    return lz_ref + k * logx + p * k * (logq - LOGQ_REF)


def main():
    d = pd.read_csv(FIT_CSV).dropna(subset=['k'])
    ids = [l.strip() for l in open(IDS) if l.strip() and not l.startswith('#')]
    d = d[d.rm_id.isin(ids)].copy()
    d['logq'] = _logQ_per_object(d.rm_id.tolist()).reindex(d.rm_id).values
    print(f'{len(d)} railed objects; median log Q - log Q_ref = '
          f'{np.median(d.logq - LOGQ_REF):+.2f}; grid log Z = [{LZ_LO:.2f}, {LZ_HI:.2f}]\n')
    for sign in (-1, 1):
        s = d[np.sign(d.k) == sign]
        print(f'--- railed at k = {sign:+d} ({len(s)} objects), median p = {s.p.median():.2f}, '
              f'median log Z_ref = {s.log10_Z_norm.median():+.2f}')
        for kk in (1, 2):
            k = sign * kk
            lo, hi, f_hi, f_lo = [], [], [], []
            for _, r in s.iterrows():
                lz = window(r.log10_Z_norm, r.p, k, r.logq)
                lo.append(lz.min()); hi.append(lz.max())
                f_hi.append(np.mean(lz > LZ_HI)); f_lo.append(np.mean(lz < LZ_LO))
            print(f'  k = {k:+d}: window log Z median from {np.median(lo):+.2f} to {np.median(hi):+.2f} '
                  f'(max {np.max(hi):+.2f}); objects reaching > 20 Zsun: {np.mean(np.array(hi) > LZ_HI):.0%}, '
                  f'< 0.05 Zsun: {np.mean(np.array(lo) < LZ_LO):.0%}; median window fraction clipped '
                  f'high {np.median(f_hi):.0%}, low {np.median(f_lo):.0%}')


if __name__ == '__main__':
    main()
