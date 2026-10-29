#!/usr/bin/env python3
"""Compare N V across the Auger / secondary-ionization flag-test Cloudy runs,
for both the strong-bump SED (flagtest_*) and the AGN SED (agntest_*).
Single solar model, n10/phi19/N24/v100."""
from pathlib import Path

BASE = Path('/Users/jiamuh/c23.01/my_models/nagao_flagtest')
SEDS = {'flagtest': 'strong-bump SED', 'agntest': 'AGN SED'}
CONFIGS = {'base': 'both on', 'noauger': 'no Auger',
           'nosecond': 'no secondary', 'noboth': 'no Auger + no secondary'}
COLS = {'N5': 'blnd 1240.00A', 'N4': 'blnd 1486.00A',
        'N3': 'blnd 1750.00A', 'C4': 'blnd 1549.00A',
        'He2': 'he 2 1640.41A'}


def load(prefix, tag):
    lines = (BASE / f'{prefix}_{tag}_lines.txt').read_text().splitlines()
    names = [c.strip().lstrip('#') for c in lines[0].split('\t')][1:]
    vals = [float(x) for x in lines[1].split('\t')[1:]]
    d = dict(zip(names, vals))
    return {k: d[v] for k, v in COLS.items()}


for prefix, sed in SEDS.items():
    data = {t: load(prefix, t) for t in CONFIGS}
    b = data['base']
    print(f'\n=== {sed} ===')
    hdr = f'{"config":<26}{"N V":>10}{"N V/CIV":>11}{"NV vs base":>12}'
    print(hdr)
    print('-' * len(hdr))
    for t, desc in CONFIGS.items():
        d = data[t]
        print(f'{desc:<26}{d["N5"]:>10.4f}{d["N5"]/d["C4"]:>11.4f}'
              f'{d["N5"]/b["N5"]:>11.3f}x')
