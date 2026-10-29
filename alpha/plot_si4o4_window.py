"""(SiIV+OIV)/CIV per-cloud map + the LOC averaging window for the 2-ratio
solution (k=0, p=0.35, Z=3.28, log r_ref=16.37, logQ=55).  Checks how far the
cloud window runs off the high-phi grid edge and whether the clipped clouds
carry any weight / emission.

    python3 -m alpha.plot_si4o4_window
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

from alpha.loc_gradient_fit import (make_interpolators, OBS_RANGE, LOGN_GRID,
                                    LOGPHI_GRID, LOGQ_REF, _LOG4PI, GAMMA_LOC,
                                    BETA_N_LOC, LOGPHI_LO_LOC, LOG_C_LIGHT,
                                    LOGU_LO_LOC, _logu_weight)

K, P, Z, LOGQ, LOG_R_REF = 0.0, 0.35, 3.28, 55.0, 16.37
OUT = 'plots/alpha/loc_gradient_noA/si4o4_window.png'


def main():
    interps = make_interpolators()
    phi_edge = LOGPHI_GRID.max()

    # --- per-cloud (SiIV+OIV)/CIV map over the Cloudy grid ---
    ng = 240
    logn = np.linspace(LOGN_GRID.min(), LOGN_GRID.max(), ng)
    logphi = np.linspace(LOGPHI_GRID.min(), LOGPHI_GRID.max(), ng)
    NN, PP = np.meshgrid(logn, logphi, indexing='ij')
    qp = np.stack([NN, PP, np.full(NN.shape, Z)], -1).reshape(-1, 3)
    si = interps['Si4'](qp) + interps['O4'](qp)
    c4 = interps['C4'](qp)
    logratio = np.log10(si / c4).reshape(NN.shape)
    logratio[(PP - NN - LOG_C_LIGHT) <= LOGU_LO_LOC] = np.nan   # remove low-ionization (log U <= -3) clouds

    # --- LOC averaging window for the solution point ---
    n_x, n_n = 80, 24
    logx = np.linspace(-1, 1, n_x)
    logn_w = np.linspace(9, 12, n_n)
    logphi_ref = LOGQ_REF - _LOG4PI - 2 * LOG_R_REF
    logphi_Q = logphi_ref - 2 * logx + (1 - 2 * P) * (LOGQ - LOGQ_REF)
    W = np.outer(10.0 ** (logn_w * (BETA_N_LOC + 1)), 10.0 ** (logx * (GAMMA_LOC + 3)))
    LN, LPH = np.meshgrid(logn_w, logphi_Q, indexing='ij')
    W_eff = W * (logphi_Q >= LOGPHI_LO_LOC)[None, :] * _logu_weight(LPH - LN - LOG_C_LIGHT)
    logphi_c = np.clip(LPH, LOGPHI_GRID.min(), phi_edge)
    qw = np.stack([LN, logphi_c, np.full(LN.shape, Z)], -1).reshape(-1, 3)
    eSi = (interps['Si4'](qw) + interps['O4'](qw)).reshape(LN.shape)
    eC4 = interps['C4'](qw).reshape(LN.shape)

    clip = LPH > phi_edge
    fW = W_eff[clip].sum() / W_eff.sum()
    fSi = (W_eff * eSi)[clip].sum() / (W_eff * eSi).sum()
    fC4 = (W_eff * eC4)[clip].sum() / (W_eff * eC4).sum()
    win = (logphi_Q.min(), logphi_Q.max())
    print(f"solution: k={K}, p={P}, Z={Z}, logQ={LOGQ}, log_r_ref={LOG_R_REF}")
    print(f"  logphi_ref = {logphi_ref:.2f}")
    print(f"  cloud window logphi = [{win[0]:.2f}, {win[1]:.2f}];  grid edge = {phi_edge:.0f}")
    print(f"  window overshoots the grid edge by {win[1] - phi_edge:.2f} dex")
    print(f"  beyond the grid edge:  LOC weight {fW:.1%},  SiIV+OIV emission {fSi:.1%},  CIV emission {fC4:.1%}")

    # --- plot ---
    fig, ax = plt.subplots(2, 1, figsize=(9.5, 9.5),
                           gridspec_kw=dict(height_ratios=[2, 1], hspace=0.07))
    lo, hi = np.floor(np.nanmin(logratio)), np.ceil(np.nanmax(logratio))
    cp = ax[0].contourf(logphi, logn, logratio, levels=np.arange(lo, hi + 0.01, 1.0),
                        cmap='viridis')
    obs, oerr = OBS_RANGE['(SiIV+OIV)/CIV']
    ax[0].contourf(logphi, logn, logratio, levels=[obs - oerr, obs + oerr],
                   colors=['cyan'], alpha=0.6, zorder=3)
    ax[0].add_patch(Rectangle((win[0], 9.0), win[1] - win[0], 3.0, fill=False,
                              ec='red', lw=3, zorder=5))
    fig.colorbar(cp, ax=ax[0], pad=0.02).set_label(
        r'$\log[\rm (Si\,IV+O\,IV)/C\,IV]$', rotation=270, labelpad=22)
    ax[0].set_ylabel(r'$\log n_{\rm H}~[\rm cm^{-3}]$')
    ax[0].legend(handles=[Patch(fc='cyan', ec='cyan', alpha=0.6, label='observed (SDSS-RM)'),
                          Patch(fc='none', ec='red', label='LOC averaging window')],
                 fontsize=11, loc='lower left')

    wpr = W_eff.sum(0)
    lSi = (W_eff * eSi).sum(0)
    lC4 = (W_eff * eC4).sum(0)
    o = np.argsort(logphi_Q)
    x = logphi_Q[o]
    ax[1].plot(x, wpr[o] / wpr.max(), color='0.4', lw=3, label=r'LOC weight $dW/d\log\Phi$')
    ax[1].plot(x, lSi[o] / lSi.max(), color='orange', lw=3, label=r'$dL_{\rm SiIV+OIV}/d\log\Phi$')
    ax[1].plot(x, lC4[o] / lC4.max(), color='purple', lw=3, label=r'$dL_{\rm CIV}/d\log\Phi$')
    ax[1].set_xlabel(r'$\log \Phi_{\rm H}~[\rm photons~cm^{-2}~s^{-1}]$')
    ax[1].set_ylabel('integrand (norm.)')
    ax[1].legend(fontsize=10, loc='upper left')

    for a in ax:
        a.axvline(phi_edge, color='red', ls='--', lw=2.2)
        a.axvspan(phi_edge, win[1] + 0.4, color='red', alpha=0.13, zorder=0)
        a.set_xlim(LOGPHI_GRID.min(), win[1] + 0.4)
        a.minorticks_on()
        a.tick_params(top=True, right=True, which='major', length=8, width=1.5, direction='in')
        a.tick_params(top=True, right=True, which='minor', length=4, width=1, direction='in')
    ax[0].text(phi_edge + 0.05, 11.6, 'off grid\n(clipped)', color='red', fontsize=11,
               va='top', ha='left')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
