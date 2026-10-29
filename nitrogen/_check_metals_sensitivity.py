#!/usr/bin/env python3
"""Debug: how do N3/N4/N5 (and CIV) respond to the overall metals scaling?

Compares the m1 (metals=1) and m10 (metals=10) Cloudy reference runs that
vary the nitrogen scale factor on the same linear grid.
"""
import numpy as np
from plot_aox_line_ratios import parse_line_list

M1 = ('/Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100/'
      'strong_n10_phi19_N24_m1_v100_LineList_BLR_Fe2.txt')
M10 = ('/Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100_m10/'
       'strong_n10_phi19_N24_m10_v100_LineList_BLR_Fe2.txt')
N_GRID = np.arange(0.1, 10.0, 0.2)

COLS = {'N3': 'blnd 1750.00A', 'N4': 'blnd 1486.00A',
        'N5': 'blnd 1240.00A', 'C4': 'blnd 1549.00A',
        'He2': 'he 2 1640.41A', 'Hb': 'H  1 4861.32A'}


def load(path):
    names, rows = parse_line_list(path)
    idx = {k: names.index(v) for k, v in COLS.items()}
    return {k: rows[:, i] for k, i in idx.items()}


m1, m10 = load(M1), load(M10)
n = min(len(m1['C4']), len(m10['C4']), len(N_GRID))
g = N_GRID[:n]

print(f'{"N/H":>6} | {"NIII]/CIV":>20} | {"NIV]/CIV":>20} | {"NV/CIV":>20}')
print(f'{"":>6} | {"m1":>9} {"m10":>9}  | {"m1":>9} {"m10":>9}  | '
      f'{"m1":>9} {"m10":>9} ')
print('-' * 78)
for i in range(0, n, 5):
    r = {}
    for ln in ('N3', 'N4', 'N5'):
        r[ln] = (m1[ln][i] / m1['C4'][i], m10[ln][i] / m10['C4'][i])
    print(f'{g[i]:6.1f} | {r["N3"][0]:9.4f} {r["N3"][1]:9.4f}  | '
          f'{r["N4"][0]:9.4f} {r["N4"][1]:9.4f}  | '
          f'{r["N5"][0]:9.4f} {r["N5"][1]:9.4f} ')

print('\n--- m10/m1 ratio of the RAW line intensities (per grid point) ---')
print(f'{"N/H":>6} | {"NIII]":>9} {"NIV]":>9} {"NV":>9} {"CIV":>9}'
      f' {"HeII":>9}')
for i in range(0, n, 5):
    print(f'{g[i]:6.1f} | '
          f'{m10["N3"][i]/m1["N3"][i]:9.3f} '
          f'{m10["N4"][i]/m1["N4"][i]:9.3f} '
          f'{m10["N5"][i]/m1["N5"][i]:9.3f} '
          f'{m10["C4"][i]/m1["C4"][i]:9.3f} '
          f'{m10["He2"][i]/m1["He2"][i]:9.3f}')

# at N/H ~ 1 (closest grid point)
j = int(np.argmin(np.abs(g - 1.0)))
print(f'\nAt N/H = {g[j]:.1f}:')
for ln, lab in (('N3', 'NIII]'), ('N4', 'NIV]'), ('N5', 'NV'),
                ('C4', 'CIV')):
    print(f'  {lab:6s} intensity  m1={m1[ln][j]:.4e}  m10={m10[ln][j]:.4e}'
          f'  (m10/m1={m10[ln][j]/m1[ln][j]:.3f})')
for ln, lab in (('N3', 'NIII]'), ('N4', 'NIV]'), ('N5', 'NV')):
    r1 = m1[ln][j] / m1['C4'][j]
    r10 = m10[ln][j] / m10['C4'][j]
    print(f'  {lab:6s}/CIV   m1={r1:.4f}  m10={r10:.4f}  '
          f'(m10/m1={r10/r1:.3f})')
