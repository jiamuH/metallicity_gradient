"""Per-cell equivalent-width contour plot for N III] 1750, N IV] 1486, N V 1240
in the (log phi, log n_H) plane, evaluated on the **alpha LOC grid** at a chosen
Z_alpha slice (default Z_alpha = 3 Z_sun, matching the m3 single-zone runs).

Recipe (matches the old Cloudy_LOC_nitrogen_old.ipynb):
    EW := line_intensity / nFnu(continuum at a nearby wavelength)
with line and continuum both read from the same line-list file (rel-to-H-beta units).

Run from repo root:
    python3 -m nitrogen.plot_loc_grid_contours
    python3 -m nitrogen.plot_loc_grid_contours 10     # for Z_alpha = 10 Z_sun
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.weight': 'heavy',
    'font.size': 16, 'axes.linewidth': 2, 'lines.linewidth': 2,
    'axes.labelweight': 'heavy', 'text.latex.preamble': r'\usepackage{bm} \usepackage{amsmath} \boldmath',
})

ALPHA_BASE   = ('/Users/jiamuh/c23.01/my_models/loc_metal/'
                'strong_LOC_varym_N25_LineList_BLR_Fe2.txt')                # phi 17-21,   Z 1-20
ALPHA_PHIEXT = ('/Users/jiamuh/c23.01/my_models/loc_metal/'
                'strong_LOC_varym_N25_phiext_LineList_BLR_Fe2.txt')         # phi 21.5-23, Z 1-20
PHI_BASE  = np.arange(17.0, 21.5, 0.5)                                       # 9
PHI_EXT   = np.arange(21.5, 23.1, 0.5)                                       # 4
PHI_AX    = np.concatenate([PHI_BASE, PHI_EXT])                              # 13
LOGN_AX   = np.arange(9.0, 12.5, 0.5)                                        # 7
Z_AX      = np.arange(1.0, 20.5, 0.5)                                        # 39 (super-solar slab)
LOG_C     = np.log10(3e10)

LINE_INFO = {
    'N3': dict(col='blnd 1750.00A', cont='nFnu 1793.44A', label=r'$\rm N\,III]\,1750$'),
    'N4': dict(col='blnd 1486.00A', cont='nFnu 1458.33A', label=r'$\rm N\,IV]\,1486$'),
    'N5': dict(col='blnd 1240.00A', cont='nFnu 1262.79A', label=r'$\rm N\,V\,1240$'),
}


def _load_alpha_file(path, n_phi):
    """Read one alpha LOC line-list file into a DataFrame with header,
    ordered (phi outer, n_H middle, Z inner)."""
    df = pd.read_csv(path, sep='\t', header=0)
    df = df[~df.iloc[:, 0].astype(str).str.contains('GRID_DELIMIT')].reset_index(drop=True)
    expected = n_phi * len(LOGN_AX) * len(Z_AX)
    assert len(df) == expected, f'{path}: {len(df)} rows != {expected}'
    return df


def _ew_at_zslice(z_alpha):
    """Return a dict {'N3','N4','N5': (n_phi=13, n_n=7) EW grid} at the requested
    Z_alpha (must be on the Z_AX grid, i.e. one of 1, 1.5, 2, ..., 20 Z_sun)."""
    i_z = int(np.argmin(np.abs(Z_AX - z_alpha)))
    z_used = Z_AX[i_z]
    df_b = _load_alpha_file(ALPHA_BASE,   len(PHI_BASE))
    df_e = _load_alpha_file(ALPHA_PHIEXT, len(PHI_EXT))
    nN, nZ = len(LOGN_AX), len(Z_AX)
    ew = {ln: np.zeros((len(PHI_AX), nN)) for ln in LINE_INFO}
    for chunk, df, p0 in [('base', df_b, 0), ('ext', df_e, len(PHI_BASE))]:
        n_phi_chunk = len(PHI_BASE) if chunk == 'base' else len(PHI_EXT)
        for jp in range(n_phi_chunk):
            for in_ in range(nN):
                idx = jp * nN * nZ + in_ * nZ + i_z
                row = df.iloc[idx]
                for ln, info in LINE_INFO.items():
                    f_line = float(row[info['col']])
                    f_cont = float(row[info['cont']])
                    ew[ln][p0 + jp, in_] = f_line / max(f_cont, 1e-30)
    return ew, z_used


def main(z_alpha=3.0):
    ew, z_used = _ew_at_zslice(z_alpha)
    print(f'Using Z_alpha = {z_used:.1f} Z_sun  (requested {z_alpha:.1f})')
    os.makedirs('plots/nitrogen', exist_ok=True)
    cbar_label = {'N3': r'$\log\,{\rm EW}\,({\rm N\,III]})$',
                  'N4': r'$\log\,{\rm EW}\,({\rm N\,IV]})$',
                  'N5': r'$\log\,{\rm EW}\,({\rm N\,V})$'}
    for ln in ('N3', 'N4', 'N5'):
        z = np.log10(np.maximum(ew[ln].T, 1e-30))                        # (n_n, n_phi)
        fig, ax = plt.subplots(figsize=(7, 6))
        cp = ax.contourf(PHI_AX, LOGN_AX, z, levels=20, cmap='viridis')
        # constant log U contours
        Pm, Nm = np.meshgrid(PHI_AX, LOGN_AX)
        cs = ax.contour(PHI_AX, LOGN_AX, Pm - Nm - LOG_C, colors='k', alpha=0.5, linestyles=':')
        ax.clabel(cs, inline=True, fontsize=14)
        # peak
        ip, in_ = np.unravel_index(np.argmax(ew[ln]), ew[ln].shape)
        ax.plot(PHI_AX[ip], LOGN_AX[in_], marker='*', markersize=28, linestyle='None',
                color='r', mec='k', mew=0.8, zorder=5)
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$n$')
        ax.set_xlim(PHI_AX.min(), PHI_AX.max())
        ax.set_ylim(LOGN_AX.min(), LOGN_AX.max())
        fig.colorbar(cp, ax=ax, label=cbar_label[ln])
        fig.tight_layout()
        out = f'plots/nitrogen/loc_grid_{ln}_contour_Zalpha{z_used:g}.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f'  {ln}: peak EW = {ew[ln].max():.3e} at phi = {PHI_AX[ip]:.1f}, n_H = {LOGN_AX[in_]:.1f}  -> {out}')


if __name__ == '__main__':
    z = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    main(z_alpha=z)
