#!/usr/bin/env python3
"""Quick check: what sets the radial extent of the population alpha profile?

Prints the LOC window constants and the distribution of the 71 objects'
breathing radii r_blr (in light-days), plus where the >=10-object coverage
window lies.
"""
import numpy as np
import pandas as pd

from alpha.plot_Z_profiles import (_drop_railed, _logQ_per_object, LOG_Q_REF,
                                   LOGX_LO, LOGX_HI, LD_PER_CM)
from plot_N_alpha_combined import CSV

df = pd.read_csv(CSV).dropna(
    subset=['k', 'log_r_ref', 'log10_Z_norm', 'z']).reset_index(drop=True)
df['logQ_med'] = _logQ_per_object(df['rm_id'].tolist()).values
df['log_r_blr'] = df['log_r_ref'] + df['p'] * (df['logQ_med'] - LOG_Q_REF)
df = _drop_railed(df)

r_ld = 10 ** df['log_r_blr'].values * LD_PER_CM
print(f'LOGX window: [{LOGX_LO}, {LOGX_HI}]  (x = r/r_Q)')
print(f'{len(df)} objects; breathing radius r_blr [ld]:')
print(f'  min {r_ld.min():7.2f}  16% {np.percentile(r_ld, 16):7.2f}  '
      f'median {np.median(r_ld):7.2f}  84% {np.percentile(r_ld, 84):7.2f}  '
      f'max {r_ld.max():7.2f}')
print(f'median logQ_med = {df["logQ_med"].median():.2f} '
      f'(Q_ref = 10^{LOG_Q_REF:.0f})')
print(f'median log r_ref [cm] = {df["log_r_ref"].median():.2f}, '
      f'median p = {df["p"].median():.2f}')
