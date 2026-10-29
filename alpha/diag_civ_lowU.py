"""Diagnostic: raw Cloudy C IV / Mg II grid values vs ionization parameter.

Checks where C IV 1549 drops to unreliably small (or zero) intensity in the
alpha LOC grid -- i.e. where the per-cloud MgII/CIV map is dominated by an
unconverged / floored CIV.  Prints, at the Z = 3 Z_sun slice, the raw line
intensities and log U = log phi - log n - log c over the (log n_H, log phi)
grid.  Uses load_cloudy_grid() so the values are RAW (no 1e-30 interp floor).

Run from repo root:
    python3 -m alpha.diag_civ_lowU
"""
import numpy as np
from alpha.loc_gradient_fit import load_cloudy_grid, LOGN_GRID, LOGPHI_GRID, Z_GRID

Z_SLICE = 3.0
LOG_C = np.log10(3e10)        # log10(c / cm s^-1)


def _table(title, M, rowvals, colvals):
    print(f"\n{title}")
    print('  logn\\phi ' + ''.join(f'{c:7.1f}' for c in colvals))
    for i, r in enumerate(rowvals):
        print(f'  {r:7.1f}  ' + ''.join(f'{M[i, j]:7.2f}' for j in range(len(colvals))))


def main():
    arr = load_cloudy_grid()                       # raw line intensities, NO 1e-30 floor
    i_z = int(np.argmin(np.abs(Z_GRID - Z_SLICE)))
    print(f"Z slice = {Z_GRID[i_z]:g} Z_sun (requested {Z_SLICE:g})")
    print(f"logn  grid: {LOGN_GRID}")
    print(f"logphi grid: {LOGPHI_GRID}")

    mg2 = arr['Mg2'][:, :, i_z]                    # (n_n, n_phi)
    c4 = arr['C4'][:, :, i_z]
    NN, PP = np.meshgrid(LOGN_GRID, LOGPHI_GRID, indexing='ij')
    logU = PP - NN - LOG_C

    print(f"\nC4  raw: min={c4.min():.3e}  max={c4.max():.3e}  "
          f"#zeros={(c4 == 0).sum()}  #(<1e-15)={(c4 < 1e-15).sum()}  /{c4.size} cells")
    print(f"Mg2 raw: min={mg2.min():.3e}  max={mg2.max():.3e}  #zeros={(mg2 == 0).sum()}")

    with np.errstate(divide='ignore'):
        log_c4 = np.log10(c4)
        log_mg2 = np.log10(mg2)
        log_ratio = log_mg2 - log_c4

    _table('log10  C IV 1549   (raw grid)', log_c4, LOGN_GRID, LOGPHI_GRID)
    _table('log10  Mg II 2798  (raw grid)', log_mg2, LOGN_GRID, LOGPHI_GRID)
    _table('log10  (MgII / CIV)', log_ratio, LOGN_GRID, LOGPHI_GRID)
    _table('log U = logphi - logn - log c', logU, LOGN_GRID, LOGPHI_GRID)

    print("\n20 lowest-U cells (raw C4 / Mg2):")
    print(f"  {'logn':>6} {'logphi':>7} {'logU':>7} {'C4':>12} {'Mg2':>12} {'log(Mg/C)':>11}")
    order = np.argsort(logU, axis=None)
    for idx in order[:20]:
        i, j = np.unravel_index(idx, logU.shape)
        print(f"  {LOGN_GRID[i]:6.1f} {LOGPHI_GRID[j]:7.1f} {logU[i, j]:7.2f} "
              f"{c4[i, j]:12.3e} {mg2[i, j]:12.3e} {log_ratio[i, j]:11.2f}")


if __name__ == '__main__':
    main()
