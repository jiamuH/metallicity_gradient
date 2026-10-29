"""Eddington ratios (Shen et al. 2019) of the fits railed at k = -1 and k = +1,
compared with the full fitted sample of 207.

Run: python3 -m alpha._check_railed_edd
"""
import numpy as np

from alpha.sdssrm_sample_properties import sample

IDS = 'fits/alpha/k_railed_ids.txt'


def describe(label, e):
    e = e.dropna()
    print(f'{label:28s} N={len(e):3d}  median log Edd {np.median(e):+.2f}, '
          f'16-84% {np.percentile(e, 16):+.2f} to {np.percentile(e, 84):+.2f}, '
          f'max {e.max():+.2f}; fraction above log Edd = -0.5: {np.mean(e > -0.5):.0%}')


def main():
    d = sample(well_constrained=False)
    ids = [l.strip() for l in open(IDS) if l.strip() and not l.startswith('#')]
    r = d[d.rm_id.isin(ids)]
    describe('all fitted', d.logEdd)
    describe('railed at k = -1', r[r.k < 0].logEdd)
    describe('railed at k = +1', r[r.k > 0].logEdd)


if __name__ == '__main__':
    main()
