"""Difference in model log(MgII/CIV) with vs without the log U <= -3 cut,
over the (log Q, log r_ref) plane (k=0, p=0.35, Z=3 Z_sun -- the c-row panel).

    python3 -m alpha.plot_logu_cut_diff
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import alpha.loc_gradient_fit as M
from alpha.loc_gradient_fit import make_interpolators, model_line_ratios, LOGU_LO_LOC

K, P, Z = 0.0, 0.35, 3.0
OUT = 'plots/alpha/loc_gradient_noA/mgii_civ_logu_cut_diff.png'


def main(n_Q=70, n_r=60, out=OUT):
    interps = make_interpolators()
    logQ = np.linspace(53.5, 56.0, n_Q)
    r_grid = np.linspace(17.0, 17.95, n_r)

    def _stack():
        m = np.zeros((n_r, n_Q))
        for j, lrr in enumerate(r_grid):
            rr, _, _ = model_line_ratios(interps, logQ, K, P, log_r_ref=lrr, z_norm=Z)
            m[j] = np.log10(rr['MgII/CIV'])
        return m

    with_cut = _stack()
    M.LOGU_LO_LOC = -np.inf
    no_cut = _stack()
    M.LOGU_LO_LOC = LOGU_LO_LOC
    diff = with_cut - no_cut

    fig, ax = plt.subplots(figsize=(8.5, 7))
    vmx = float(np.nanmax(np.abs(diff)))
    im = ax.pcolormesh(logQ, r_grid, diff, shading='auto', cmap='RdBu_r',
                       vmin=-vmx, vmax=vmx)
    levels = np.unique(np.round(np.linspace(-vmx, vmx, 9)[1:-1], 2))
    if np.size(levels):
        cs = ax.contour(logQ, r_grid, diff, colors='k', linewidths=0.9, levels=levels)
        ax.clabel(cs, fmt='%.2f', fontsize=13, inline=True)

    _bb = dict(boxstyle='round,pad=0.25', fc='0.12', ec='none', alpha=0.6)
    ax.text(0.03, 0.96, rf'$k={K:.2f},\ p={P:.2f},\ Z_{{\rm norm}}={Z:g}\,Z_{{\odot}}$',
            transform=ax.transAxes, va='top', ha='left', color='w', bbox=_bb)
    ax.set_xlabel(r'$\log Q~[\rm photons~s^{-1}]$')
    ax.set_ylabel(r'$\log r_{\rm ref}$')
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r'$\Delta \log(\rm Mg\,II\,/\,C\,IV)$  (cut $-$ no cut)',
                 rotation=270, labelpad=24)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True,
                   length=8, width=1.5, direction='in', pad=5)
    ax.tick_params(axis='both', which='minor', top=True, right=True,
                   length=4, width=1, direction='in')

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}  (diff range {diff.min():.3f} .. {diff.max():.3f} dex)")


if __name__ == '__main__':
    main()
