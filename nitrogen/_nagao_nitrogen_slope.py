"""Nitrogen abundance and two-point (N V -> N IV]) radial slope for the SDSS
quasar composites of Nagao et al. (2006a, Table 3, 2.0 <= z < 2.5, five
absolute-magnitude bins), compared with the same two-point slope for the
Constantin & Shields (2003) NLS1 composite (referee comment 4 check).

Abundances use the same LOC inversion as Section 2.2 (infer, Z_alpha = 3) and
the same LOC emission-weighted radii.  Nagao et al. fluxes are relative to
C IV = 1 with statistical errors only.

Run (from the project root): python3 nitrogen/_nagao_nitrogen_slope.py
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from loc_nitrogen_abundance import CSH, consistent_loc_radii   # noqa: E402
from plot_loc_nitrogen_composites import infer                  # noqa: E402

NAGAO_TEX = ('/private/tmp/claude-501/-Users-jiamuh-python-metallicity-gradient/'
             '1bb174d7-7361-425a-9ae9-bacf96eb632a/scratchpad/arx_0510385')
BINS = ['-24.5>M_B>=-25.5', '-25.5>M_B>=-26.5', '-26.5>M_B>=-27.5',
        '-27.5>M_B>=-28.5', '-28.5>M_B>=-29.5']
ROWS = {'N5': r'N{\sc v}$\lambda$1240', 'N4': r'N{\sc iv}]$\lambda$1486',
        'C4': r'C{\sc iv}$\lambda$1549', 'O3': r'O{\sc iii}]$\lambda$1663'}


def nagao_table3():
    """{bin: {'N5': (flux, err), ...}} from Nagao et al. (2006a) Table 3."""
    tex = [f for f in os.listdir(NAGAO_TEX) if f.endswith('.tex')][0]
    lines = open(os.path.join(NAGAO_TEX, tex)).read().splitlines()
    start = next(i for i, l in enumerate(lines) if 'label{table:03}' in l)
    out = {b: {} for b in BINS}
    for key, name in ROWS.items():
        i = next(j for j in range(start, start + 120) if lines[j].startswith(name))
        vals = []
        for l in lines[i + 1:i + 6]:
            m = re.search(r'([\d.]+)\s*\$\\pm\$\s*([\d.]+)', l)
            vals.append((float(m.group(1)), float(m.group(2))) if m else None)
        for b, v in zip(BINS, vals):
            if v is not None:
                out[b][key] = v
    return out


def two_point(src, radii, den='C4'):
    """X for N V and N IV] and the slope dlogX/dlogr between their radii."""
    res = {}
    for ion in ('N5', 'N4'):
        res[ion] = infer(ion, src, den)          # (X, lo, hi, extrapolated)
    dlr = np.log10(radii['N4'][0]) - np.log10(radii['N5'][0])
    slope = (np.log10(res['N4'][0]) - np.log10(res['N5'][0])) / dlr
    return res, slope


def main():
    radii = consistent_loc_radii()
    print(f"radii: N V {radii['N5'][0]:.1f} ld, N IV] {radii['N4'][0]:.1f} ld\n")
    rows = [('C&S03 NLS1', CSH)] + [(f'Nagao {b}', d) for b, d in nagao_table3().items()]
    for den in ('C4', 'O3'):
        print(f'--- denominator {den} ---')
        for label, src in rows:
            if not all(k in src for k in ('N5', 'N4', den)):
                print(f'{label:28s} missing a line')
                continue
            res, slope = two_point(src, radii, den)
            fmt = lambda r: f"{r[0]:6.2f} [{r[1]:.2f},{r[2]:.2f}]{'*' if r[3] else ' '}"
            print(f"{label:28s} N5/{den}={src['N5'][0] / src[den][0]:.3f} "
                  f"N4/{den}={src['N4'][0] / src[den][0]:.4f}  "
                  f"X(N V)={fmt(res['N5'])} X(N IV])={fmt(res['N4'])}  slope={slope:+.2f}")
        print('(* = beyond the model grid, power-law extrapolation)\n')


if __name__ == '__main__':
    main()
