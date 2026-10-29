"""Debug: why N III] emission extends to higher ionizing flux at Z_alpha = 3 Z_sun than at Z_sun.
Compares, model by model (same log phi, log n_H), the radius-averaged nitrogen ion fractions and
the mean electron temperature where N2+ and N3+ live, read from the concatenated grid output
(strong_LOC_m1_N23.out vs strong_LOC_m3_N23.out; blocks are in grid-index order).

Run: python3 nitrogen/_check_n3_zone_vs_metals.py
"""
import os
import re

import numpy as np

DIR = os.path.expanduser('~/c23.01/my_models/loc_metal_nitrogen')
NUM = re.compile(r'-?\d+\.\d+')


def grid_params(prefix):
    rows = [l.split('\t') for l in open(os.path.join(DIR, prefix + '.grd')) if not l.startswith('#')]
    return [(float(r[6]), float(r[7])) for r in rows]


def blocks(prefix):
    """Per model: log mean N ion fractions (over radius) and log mean T of each N ion."""
    out, cur = [], None
    for line in open(os.path.join(DIR, prefix + '.out'), errors='replace'):
        if 'Log10 Mean Ionisation (over radius)' in line and 'electron' not in line:
            cur = 'ion'
        elif 'Log10 Mean Temperature (over radius)' in line and 'electron' not in line:
            cur = 'T'
        elif 'Log10 Mean' in line:
            cur = None
        elif cur and line.lstrip().startswith('Nitrogen'):
            vals = [float(v) for v in NUM.findall(line)]
            if cur == 'ion':
                out.append({'ion': vals})
            else:
                out[-1]['T'] = vals
            cur = None
    return out


if __name__ == '__main__':
    p1, p3 = grid_params('strong_LOC_m1_N23'), grid_params('strong_LOC_m3_N23')
    assert p1 == p3
    b1, b3 = blocks('strong_LOC_m1_N23'), blocks('strong_LOC_m3_N23')
    print(f'{len(b1)} and {len(b3)} models parsed')
    print('log phi  log n | log f(N2+)  Z_sun / 3Z_sun | log f(N3+)  Z_sun / 3Z_sun | '
          'log T(N2+)  Z_sun / 3Z_sun | log T(N3+)  Z_sun / 3Z_sun')
    for (phi, n), a, b in zip(p1, b1, b3):
        if phi in (18.0, 18.5, 19.0, 19.5, 20.0) and n in (10.0, 11.0):
            print(f'{phi:6.1f} {n:6.1f}  |   {a["ion"][2]:6.2f}  {b["ion"][2]:6.2f}         |'
                  f'   {a["ion"][3]:6.2f}  {b["ion"][3]:6.2f}         |'
                  f'   {a["T"][2]:5.2f}  {b["T"][2]:5.2f}           |   {a["T"][3]:5.2f}  {b["T"][3]:5.2f}')
