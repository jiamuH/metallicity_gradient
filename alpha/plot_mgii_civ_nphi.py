"""2D map of the model MgII/CIV ratio in the (n_H, phi) plane at fixed Z.

Raw per-cloud Cloudy line ratio (no LOC integral): log(MgII/CIV) over the
(log phi, log n_H) plane at Z = 3 Z_sun, drawn as a filled-contour (contourf)
map in the same style as the nitrogen EW plots (1-dex bands, viridis, dotted
constant-log-U overlay), with the BFKV95 observed ratio range shaded.

Run from repo root:
    python3 -m alpha.plot_mgii_civ_nphi
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from alpha.loc_gradient_fit import (make_interpolators, MGII_SCALE, OBS_RANGE,
                                    LOGN_GRID, LOGPHI_GRID, LOGU_LO_LOC)

Z_NORM = 3.0
LOG_C = np.log10(3e10)        # log10(c / cm s^-1); for the constant-log-U overlay
OUT = 'plots/alpha/loc_gradient_noA/mgii_civ_nphi_Z3.png'


def plot_mgii_civ_nphi(z_norm=Z_NORM, n_grid=240, out=OUT):
    """Per-cloud log(MgII/CIV) over (log phi, log n_H) at fixed Z, as a filled-
    contour map (same contourf style as the nitrogen EW plots), with the BFKV95
    observed ratio band shaded and constant-log-U lines overlaid.  Uses the
    trilinear Cloudy interpolators (no LOC integration), so the map is the raw
    single-cloud emissivity ratio."""
    interps = make_interpolators()
    logn = np.linspace(LOGN_GRID.min(), LOGN_GRID.max(), n_grid)
    logphi = np.linspace(LOGPHI_GRID.min(), LOGPHI_GRID.max(), n_grid)
    NN, PP = np.meshgrid(logn, logphi, indexing='ij')          # (n_logn, n_logphi)
    qpts = np.stack([NN, PP, np.full(NN.shape, z_norm)], axis=-1).reshape(-1, 3)
    mg2 = interps['Mg2'](qpts).reshape(NN.shape)
    c4 = interps['C4'](qpts).reshape(NN.shape)
    logratio = np.log10(MGII_SCALE * mg2 / c4)
    logratio[(PP - NN - LOG_C) <= LOGU_LO_LOC] = np.nan   # remove low-ionization (log U <= -3) clouds

    fig, ax = plt.subplots(figsize=(8.5, 7))
    # filled contours in 1-dex bands (same contourf style as the nitrogen EW maps)
    lo = float(np.floor(np.nanmin(logratio)))
    hi = float(np.ceil(np.nanmax(logratio)))
    levels = np.arange(lo, hi + 0.01, 1.0)
    cp = ax.contourf(logphi, logn, logratio, levels=levels, cmap='viridis')

    # BFKV95 observed range: shade where the model log(MgII/CIV) lies within
    # the observed ratio band (center +/- half-width)
    obs, oerr = OBS_RANGE['MgII/CIV']
    ax.contourf(logphi, logn, logratio, levels=[obs - oerr, obs + oerr],
                colors=['cyan'], alpha=0.55, zorder=3)
    ax.contour(logphi, logn, logratio, levels=[obs - oerr, obs + oerr],
               colors='cyan', linewidths=1.5, zorder=4)

    # constant log U overlay (U = phi / n_H c), dotted black, as in the nitrogen plots
    Pm, Nm = np.meshgrid(logphi, logn)
    cs = ax.contour(logphi, logn, Pm - Nm - LOG_C, colors='k', alpha=0.5,
                    linestyles=':', zorder=5)
    ax.clabel(cs, inline=True, fontsize=13)

    obs_proxy = Patch(facecolor='cyan', edgecolor='cyan', alpha=0.55,
                      label=r'observed (BFKV95)')
    ax.legend(handles=[obs_proxy], loc='lower left', fontsize=12,
              frameon=True, facecolor='0.12', edgecolor='none', labelcolor='w')

    _bb = dict(boxstyle='round,pad=0.25', fc='0.12', ec='none', alpha=0.6)
    ax.text(0.03, 0.96, rf'$Z = {z_norm:g}\,Z_{{\odot}}$', transform=ax.transAxes,
            va='top', ha='left', color='w', bbox=_bb)

    ax.set_xlabel(r'$\log \Phi_{\rm H}~[\rm photons~cm^{-2}~s^{-1}]$')
    ax.set_ylabel(r'$\log n_{\rm H}~[\rm cm^{-3}]$')
    ax.set_xlim(logphi.min(), logphi.max())
    ax.set_ylim(logn.min(), logn.max())
    cb = fig.colorbar(cp, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label(r'$\log(\rm Mg\,II\,/\,C\,IV)$', rotation=270, labelpad=20)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True,
                   length=8, width=1.5, direction='in', pad=5)
    ax.tick_params(axis='both', which='minor', top=True, right=True,
                   length=4, width=1, direction='in')

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == '__main__':
    plot_mgii_civ_nphi()
