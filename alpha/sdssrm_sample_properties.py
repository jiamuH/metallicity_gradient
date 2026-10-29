"""Redshift, luminosity, black-hole mass and Eddington ratio of the 71 SDSS-RM
quasars that enter the population alpha-element profile (Section 3), from the
Shen et al. (2019) SDSS-RM catalog (VizieR J/ApJS/241/34).

Catalog quantities used
  zsys      systemic redshift
  logL1350  continuum luminosity at 1350 A [erg/s]
  logLBol   bolometric luminosity [erg/s] (Richards et al. 2006 corrections)
  logBH     fiducial single-epoch black-hole mass [M_sun]
  logEdd    Eddington ratio from logBH

The catalog is fetched once and cached in data/alpha/shen2019_sdssrm.tsv.

Run: python3 -m alpha.sdssrm_sample_properties   (from the project root)
"""
import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plot_Z_profiles import _drop_railed   # noqa: E402

FIT_CSV = 'fits/alpha/loc_gradient_noA_Qref58/loc_fit_summary.csv'
CACHE = 'data/alpha/shen2019_sdssrm.tsv'
COLS = ['RMID', 'zsys', 'logL1350', 'logL5100', 'logLBol', 'logBH', 'logEdd']
URL = ('https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/ApJS/241/34/catalog'
       '&-out.max=unlimited&-out=' + ','.join(COLS))


def load_catalog():
    if not os.path.exists(CACHE):
        with urllib.request.urlopen(URL, timeout=60) as f:
            text = f.read().decode('latin-1')
        rows = [l for l in text.splitlines() if l and not l.startswith('#')]
        # rows[0] = header, rows[1] = units, rows[2] = dashes
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, 'w') as fh:
            fh.write('\n'.join([rows[0]] + rows[3:]) + '\n')
    return pd.read_csv(CACHE, sep='\t', na_values=['', ' ']).apply(pd.to_numeric, errors='coerce')


def sample(well_constrained=True):
    """The 71 well-constrained fits (or all 207 fitted quasars), joined to the
    Shen et al. (2019) catalog."""
    fits = pd.read_csv(FIT_CSV).dropna(subset=['k', 'log_r_ref', 'log10_Z_norm', 'z'])
    fits = fits.reset_index(drop=True)
    if well_constrained:
        fits = _drop_railed(fits)
    fits['RMID'] = fits['rm_id'].str[2:].astype(int)
    return fits[['rm_id', 'RMID', 'k', 'k_lo', 'k_hi']].merge(load_catalog(), on='RMID', how='left')


def main():
    df = sample()
    print(f'{len(df)} objects')
    for col in COLS[1:]:
        v = df[col].dropna().values
        if not len(v):
            print(f'{col:9s} N=  0')
            continue
        print(f'{col:9s} N={len(v):3d}  range {v.min():6.2f} to {v.max():6.2f}  median {np.median(v):6.2f}')


if __name__ == '__main__':
    main()
