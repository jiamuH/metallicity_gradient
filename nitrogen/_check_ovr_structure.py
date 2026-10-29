#!/usr/bin/env python3
"""Debug: column-averaged Te and ionization structure, m1 vs m10.

Parses the Cloudy 'save overview' grid output and reports, for the
nitrogen grid point nearest N/H = 1, the H-column-weighted mean Te and
the column-integrated ion densities of C and O.
"""
import numpy as np

OVR = {
    'm1':  ('/Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100/'
            'strong_n10_phi19_N24_m1_v100_blr.ovr'),
    'm10': ('/Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100_m10/'
            'strong_n10_phi19_N24_m10_v100_blr.ovr'),
}
# 0-indexed columns
C = dict(depth=0, Te=1, hden=3, eden=4,
         HI=6, HII=7, HeIII=10, C2=13, C3=14, C4=15,
         O3=18, O4=19, O5=20, O6=21)
GRID_IDX = 4   # N_GRID[4] = 0.9, nearest to N/H = 1


def blocks(path):
    """Return list of (Nzone, Ncol) arrays, one per grid point."""
    out, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') and 'depth' in line:
                continue
            if 'GRID_DELIMIT' in line:
                if cur:
                    out.append(np.array(cur))
                cur = []
                continue
            parts = line.split()
            if len(parts) > 10:
                try:
                    cur.append([float(x) for x in parts])
                except ValueError:
                    pass
    if cur:
        out.append(np.array(cur))
    return out


def analyze(path):
    blk = blocks(path)[GRID_IDX]
    depth = blk[:, C['depth']]
    dl = np.diff(depth, prepend=0.0)          # zone thickness [cm]
    hden = blk[:, C['hden']]
    Te = blk[:, C['Te']]
    dNH = hden * dl                            # H column per zone
    NH = dNH.sum()
    Te_mean = np.sum(Te * dNH) / NH            # H-column-weighted Te
    res = {'NH': NH, 'Te_mean': Te_mean,
           'Te_min': Te.min(), 'Te_max': Te.max()}
    for ion in ('C2', 'C3', 'C4', 'O3', 'O4', 'O5', 'O6'):
        res[ion] = np.sum(blk[:, C[ion]] * dNH)   # ion column / X_elem
    return res


for tag, path in OVR.items():
    r = analyze(path)
    print(f'\n=== {tag} (grid pt {GRID_IDX}) ===')
    print(f'  total H column  N_H = {r["NH"]:.3e} cm^-2')
    print(f'  Te: column-mean = {r["Te_mean"]:.0f} K   '
          f'(min {r["Te_min"]:.0f}, max {r["Te_max"]:.0f})')
    print(f'  ion columns / X_elem  [cm^-2]:')
    for ion in ('C2', 'C3', 'C4', 'O3', 'O4', 'O5', 'O6'):
        print(f'    {ion:4s} {r[ion]:.3e}')

r1 = analyze(OVR['m1'])
r10 = analyze(OVR['m10'])
print('\n=== m10 / m1 ===')
print(f'  Te column-mean ratio = {r10["Te_mean"]/r1["Te_mean"]:.3f}')
print('  ion-column ratio (m10/m1), per unit element abundance:')
for ion in ('C2', 'C3', 'C4', 'O3', 'O4', 'O5', 'O6'):
    print(f'    {ion:4s} {r10[ion]/r1[ion]:.3f}')
