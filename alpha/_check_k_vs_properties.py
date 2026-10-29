"""Does the alpha-element gradient slope k of the 71 well-constrained SDSS-RM
fits correlate with black-hole mass, bolometric luminosity, Eddington ratio or
redshift? (referee comment 4 check)

Run: python3 -m alpha._check_k_vs_properties
"""
import numpy as np
from scipy.stats import spearmanr

from alpha.sdssrm_sample_properties import sample

df = sample()
print(f'{len(df)} objects, median k = {np.median(df.k):.2f}')
for col, name in (('logBH', 'black-hole mass'), ('logLBol', 'bolometric luminosity'),
                  ('logEdd', 'Eddington ratio'), ('zsys', 'redshift')):
    d = df[['k', col]].dropna()
    rho, pval = spearmanr(d[col], d['k'])
    lo, hi = np.percentile(d[col], [0, 100])
    # median k in the lower and upper halves of the property
    med = np.median(d[col])
    k_lo, k_hi = np.median(d.k[d[col] <= med]), np.median(d.k[d[col] > med])
    print(f'{name:22s} N={len(d):3d} range {lo:6.2f}-{hi:6.2f}  Spearman rho={rho:+.2f} p={pval:.2f}'
          f'  median k below/above median: {k_lo:+.2f} / {k_hi:+.2f}')
