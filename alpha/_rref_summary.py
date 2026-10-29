#!/usr/bin/env python3
"""Quick look at the fitted r_ref distribution vs what Q_ref implies."""
import numpy as np
import pandas as pd

CSV = 'fits/alpha/loc_gradient_noA_Qref58/loc_fit_summary.csv'
LD = 2.592e15          # 1 light-day in cm
LOG_C_Q = -13.864      # log C_Q (Nagao SED), from loc_gradient_fit.py
LOGQ_REF = 58.0

df = pd.read_csv(CSV)
lr = df['log_r_ref'].dropna().values
print(f'N objects = {len(lr)}')
for q in (0, 16, 50, 84, 100):
    print(f'  log_r_ref  {q:3d}th pct = {np.percentile(lr, q):.3f}'
          f'   ({10**np.percentile(lr, q)/LD:7.1f} ld)')
print(f'  within 0.1 dex of prior floor 17.0: '
      f'{np.sum(lr < 17.1)}/{len(lr)}')
print(f'  prior on log_r_ref is [17.0, 19.5]')

# luminosity that Q_ref = 1e58 corresponds to
logL1350 = LOGQ_REF + LOG_C_Q + np.log10(1350.0)
print(f'\nQ_ref = 1e58 s^-1  ->  lambda L_1350 = 10^{logL1350:.2f} erg/s')

# rough R-L expectations at that luminosity
# Bentz+2013 Hbeta:  log(R/ld) = 1.527 + 0.533 log(lamL5100/1e44)
for name, logL in [('lamL1350', logL1350), ('lamL5100~L1350-0.5dex', logL1350-0.5)]:
    R = 10**(1.527 + 0.533*(logL - 44.0))
    print(f'  Bentz+13 Hbeta R-L at {name}=10^{logL:.2f}: R ~ {R:.0f} ld')
