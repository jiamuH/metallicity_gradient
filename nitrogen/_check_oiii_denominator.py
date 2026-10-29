#!/usr/bin/env python3
"""Quick check: what do N_x/O III] ratios give, compared with N_x/C IV?

Reads the same single-zone Cloudy nitrogen grids used by
compute_nitrogen_abundance.py (nitrogen scale factor 0.1..9.9 in 0.2 steps,
at Z_alpha = 1 and 10 Zsun) and prints the model ratios at solar nitrogen,
plus the nitrogen scale factor at which each ratio matches a target value.

No files written; diagnostic only.
"""
import numpy as np

from compute_nitrogen_abundance import MODEL_FILES, N_GRID
from plot_aox_line_ratios import parse_line_list

COLS = {
    'N5': 'blnd 1240.00A',
    'N4': 'blnd 1486.00A',
    'N3': 'blnd 1750.00A',
    'C4': 'blnd 1549.00A',
    'O3': 'blnd 1666.00A',
    'C3': 'blnd 1909.00A',
}


def load(path):
    col_names, rows = parse_line_list(path)
    idx = {name: col_names.index(name) for name in COLS.values()}
    return {k: rows[:, idx[v]] for k, v in COLS.items()}


def solar_index():
    """Index of the grid point closest to nitrogen scale factor 1.0."""
    return int(np.argmin(np.abs(N_GRID - 1.0)))


def invert(ratio_curve, target):
    """Nitrogen scale factor where the (monotonic) model curve hits target."""
    n = min(len(ratio_curve), len(N_GRID))
    x, y = N_GRID[:n], ratio_curve[:n]
    if target <= y.min() or target >= y.max():
        return np.nan
    return float(np.interp(target, y, x))


def main():
    for tag, path in MODEL_FILES.items():
        try:
            d = load(path)
        except (IndexError, RuntimeError, ValueError):
            print(f'\n=== {tag}: NO DATA in {path} (header only) ===')
            continue
        i = solar_index()
        print(f'\n=== {tag} (nitrogen scale factor {N_GRID[i]:.1f}) ===')
        for num in ('N3', 'N4', 'N5'):
            r_civ = d[num] / d['C4']
            r_o3 = d[num] / d['O3']
            print(f'  {num}/CIV  = {r_civ[i]:.4f}    {num}/OIII] = {r_o3[i]:.4f}')
        print(f"  OIII]/CIV = {(d['O3'] / d['C4'])[i]:.4f}"
              f"   OIII]/CIII] = {(d['O3'] / d['C3'])[i]:.4f}")

        # 2023 reference points
        r_n4_civ = d['N4'] / d['C4']
        r_n3_o3 = d['N3'] / d['O3']
        print('  --- inversions ---')
        print(f"  NIV]/CIV  = 0.023 (Nagao avg)  -> N = {invert(r_n4_civ, 0.023):.2f}")
        print(f"  NIV]/CIV  = 0.0102 (VdB01)     -> N = {invert(r_n4_civ, 0.0102):.2f}")
        print(f"  NIII]/OIII] = 0.72 (D&W-E00)   -> N = {invert(r_n3_o3, 0.72):.2f}")
        print(f"  NIII]/OIII] = 0.49 (C&S03)     -> N = {invert(r_n3_o3, 0.49):.2f}")


if __name__ == '__main__':
    main()
