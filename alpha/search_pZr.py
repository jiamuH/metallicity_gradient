"""Can a (Z_norm, log r_ref) combination reproduce BOTH observed ratios at once?
k=0, p=0.35, logQ=55.  (p is degenerate with log_r_ref at fixed k, logQ.)

    python3 -m alpha.search_pZr
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from alpha.loc_gradient_fit import make_interpolators, model_line_ratios, OBS_RANGE

K, P, LOGQ = 0.0, 0.35, 55.0
OUT = 'plots/alpha/loc_gradient_noA/search_pZr.png'


def main():
    interps = make_interpolators()
    obs_mg = OBS_RANGE['MgII/CIV'][0]
    obs_si = OBS_RANGE['(SiIV+OIV)/CIV'][0]
    Z = np.logspace(np.log10(0.3), np.log10(8.0), 60)
    R = np.linspace(16.3, 18.3, 60)
    MG = np.zeros((len(R), len(Z)))
    SI = np.zeros((len(R), len(Z)))
    for ir, lrr in enumerate(R):
        for iz, z in enumerate(Z):
            rr, _, _ = model_line_ratios(interps, LOGQ, K, P, log_r_ref=lrr, z_norm=z)
            MG[ir, iz] = np.log10(rr['MgII/CIV'][0])
            SI[ir, iz] = np.log10(rr['(SiIV+OIV)/CIV'][0])

    resid = np.hypot(MG - obs_mg, SI - obs_si)
    ir, iz = np.unravel_index(np.argmin(resid), resid.shape)
    print(f"observed: MgII/CIV={obs_mg:+.3f}  (SiIV+OIV)/CIV={obs_si:+.3f}  (k={K}, p={P}, logQ={LOGQ})")
    print(f"best (Z, log_r_ref): Z={Z[iz]:.2f} Z_sun,  log_r_ref={R[ir]:.2f}")
    print(f"  model MgII/CIV       = {MG[ir, iz]:+.3f}  (resid {MG[ir, iz]-obs_mg:+.3f})")
    print(f"  model (SiIV+OIV)/CIV = {SI[ir, iz]:+.3f}  (resid {SI[ir, iz]-obs_si:+.3f})")
    print(f"  combined residual    = {resid[ir, iz]:.3f} dex")

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.pcolormesh(Z, R, resid, shading='auto', cmap='magma_r')
    fig.colorbar(im, ax=ax, pad=0.02).set_label(
        r'combined residual [dex]', rotation=270, labelpad=22)
    ax.contour(Z, R, MG, levels=[obs_mg], colors='cyan', linewidths=3)
    ax.contour(Z, R, SI, levels=[obs_si], colors='lime', linewidths=3)
    ax.plot(Z[iz], R[ir], '*', ms=26, color='white', mec='k', mew=1.5, zorder=5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$Z_{\rm norm}~[Z_\odot]$')
    ax.set_ylabel(r'$\log r_{\rm ref}$')
    ax.legend(handles=[Line2D([], [], color='cyan', lw=3, label=r'$\rm Mg\,II/C\,IV =$ obs'),
                       Line2D([], [], color='lime', lw=3, label=r'$\rm (Si\,IV+O\,IV)/C\,IV =$ obs'),
                       Line2D([], [], marker='*', color='white', mec='k', ls='', ms=16,
                              label='min residual')],
              fontsize=12, loc='upper right')
    ax.text(0.03, 0.97, rf'$k={K:g},\ p={P:g},\ \log Q={LOGQ:g}$', transform=ax.transAxes,
            va='top', bbox=dict(boxstyle='round,pad=0.3', fc='0.9', ec='none', alpha=0.85))
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, which='major', length=8, width=1.5, direction='in')
    ax.tick_params(top=True, right=True, which='minor', length=4, width=1, direction='in')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
