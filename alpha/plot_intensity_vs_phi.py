"""Per-cloud line intensity vs ionizing flux: raw Cloudy LOC grid (Z=3 Z_sun).
MgII 2798 and CIV 1549 vs log phi, one curve per density.

    python3 -m alpha.plot_intensity_vs_phi
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from alpha.loc_gradient_fit import load_cloudy_grid, LOGN_GRID, LOGPHI_GRID, Z_GRID

Z_SLICE = 3.0
OUT = 'plots/alpha/loc_gradient_noA/intensity_vs_phi_Z3.png'


def main():
    arr = load_cloudy_grid()
    i_z = int(np.argmin(np.abs(Z_GRID - Z_SLICE)))
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    cmap = plt.colormaps['plasma']
    for ax, key, ttl in [(axes[0], 'Mg2', r'$\rm Mg\,II\,2798$'),
                         (axes[1], 'C4', r'$\rm C\,IV\,1549$')]:
        grid = arr[key][:, :, i_z]                       # (n_n, n_phi)
        for i, ln in enumerate(LOGN_GRID):
            c = cmap(i / (len(LOGN_GRID) - 1))
            ax.plot(LOGPHI_GRID, grid[i], '-o', lw=2.5, ms=5, color=c,
                    label=rf'$\log n_{{\rm H}}={ln:g}$')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\log \Phi_{\rm H}~[\rm photons~cm^{-2}~s^{-1}]$')
        ax.set_ylabel(rf'intensity  {ttl}  (rel. units)')
        ax.text(0.04, 0.96, ttl + rf'$,\ Z={Z_SLICE:g}\,Z_{{\odot}}$',
                transform=ax.transAxes, va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='0.9', ec='none', alpha=0.85))
        ax.minorticks_on()
        ax.tick_params(top=True, right=True, which='major', length=8, width=1.5, direction='in')
        ax.tick_params(top=True, right=True, which='minor', length=4, width=1, direction='in')
    axes[1].legend(fontsize=11, loc='lower right')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
