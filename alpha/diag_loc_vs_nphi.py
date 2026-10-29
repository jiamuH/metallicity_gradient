"""Debug: why the LOC-averaged MgII/CIV differs from the per-cloud n-phi map.

Decomposes the LOC integral at one model point by ionization parameter log U:
the per-cloud MgII/CIV vs log U for the clouds actually in the integral, and
the LOC-weighted MgII / CIV emission distributions dL/dlog U.

    python3 -m alpha.diag_loc_vs_nphi
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from alpha.loc_gradient_fit import (make_interpolators, LOGQ_REF, _LOG4PI,
                                    LOGPHI_GRID, GAMMA_LOC, BETA_N_LOC,
                                    LOGPHI_LO_LOC, LOG_C_LIGHT, MGII_SCALE,
                                    _logu_weight, OBS_RANGE)

# model point to debug (k=0, p=0.35, Z=3 -- the c-row of the heatmap)
K, P, Z, LOGQ, LOG_R_REF = 0.0, 0.35, 3.0, 55.0, 17.5
OUT = 'plots/alpha/loc_gradient_noA/diag_loc_vs_nphi.png'


def main():
    interps = make_interpolators()
    n_x, n_n = 80, 24
    logx = np.linspace(-1.0, 1.0, n_x)
    logn = np.linspace(9.0, 12.0, n_n)
    logphi_ref = LOGQ_REF - _LOG4PI - 2.0 * LOG_R_REF
    dlq = LOGQ - LOGQ_REF
    logphi_Q = logphi_ref - 2.0 * logx + (1.0 - 2.0 * P) * dlq            # (n_x,)
    W = np.outer(10.0 ** (logn * (BETA_N_LOC + 1.0)),
                 10.0 ** (logx * (GAMMA_LOC + 3.0)))                       # (n_n, n_x)
    LN, LPH = np.meshgrid(logn, logphi_Q, indexing='ij')                   # (n_n, n_x)
    logU = LPH - LN - LOG_C_LIGHT
    W_eff = W * (logphi_Q >= LOGPHI_LO_LOC)[None, :] * _logu_weight(logU)
    logphi_c = np.clip(logphi_Q, LOGPHI_GRID.min(), LOGPHI_GRID.max())
    qpts = np.stack([LN, np.broadcast_to(logphi_c, LN.shape),
                     np.full(LN.shape, Z)], axis=-1).reshape(-1, 3)
    eMg = interps['Mg2'](qpts).reshape(LN.shape)
    eC4 = interps['C4'](qpts).reshape(LN.shape)
    rho = MGII_SCALE * eMg / eC4                                           # per-cloud ratio
    L_Mg, L_C4 = np.sum(W_eff * eMg), np.sum(W_eff * eC4)
    loc_log = np.log10(MGII_SCALE * L_Mg / L_C4)

    dMg, dC4, U = (W_eff * eMg).ravel(), (W_eff * eC4).ravel(), logU.ravel()
    band = (U >= -1.5) & (U <= -0.5)
    obs, oerr = OBS_RANGE['MgII/CIV']
    print(f"point: k={K}, p={P}, Z={Z}, logQ={LOGQ}, log_r_ref={LOG_R_REF}")
    print(f"  cloud log U span                   : [{U.min():+.2f}, {U.max():+.2f}]")
    print(f"  LOC-averaged log(MgII/CIV)         : {loc_log:+.3f}")
    print(f"  SDSS-RM observed log(MgII/CIV)     : {obs:+.3f}")
    print(f"  emission-weighted <logU>  MgII     : {np.sum(U*dMg)/dMg.sum():+.2f}")
    print(f"  emission-weighted <logU>  CIV      : {np.sum(U*dC4)/dC4.sum():+.2f}")
    print(f"  frac of L_MgII from logU in [-1.5,-0.5]: {dMg[band].sum()/dMg.sum():.3f}")
    print(f"  frac of L_CIV  from logU in [-1.5,-0.5]: {dC4[band].sum()/dC4.sum():.3f}")

    fig, ax = plt.subplots(2, 1, figsize=(9, 9.5), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08))
    sc = ax[0].scatter(U, np.log10(rho).ravel(),
                       c=np.log10(W_eff.ravel() / W_eff.max() + 1e-30),
                       s=16, cmap='viridis', vmin=-4, vmax=0)
    ax[0].axhspan(obs - oerr, obs + oerr, color='cyan', alpha=0.4, zorder=0)
    ax[0].axhline(loc_log, color='r', lw=3, label=rf'LOC average $= {loc_log:+.2f}$')
    ax[0].axhline(obs, color='c', lw=2.5, ls='--', label=rf'SDSS-RM mean $= {obs:+.2f}$')
    ax[0].set_ylabel(r'$\log(\rm Mg\,II\,/\,C\,IV)$')
    ax[0].legend(fontsize=12, loc='lower left')
    fig.colorbar(sc, ax=ax[0]).set_label(r'$\log\,(W_{\rm eff}/W_{\rm eff}^{\rm max})$',
                                         rotation=270, labelpad=22)
    _bb = dict(boxstyle='round,pad=0.3', fc='0.9', ec='none', alpha=0.85)
    ax[0].text(0.03, 0.96, rf'$k={K:g},\ p={P:g},\ Z={Z:g}\,Z_{{\odot}},\ '
               rf'\log Q={LOGQ:g},\ \log r_{{\rm ref}}={LOG_R_REF:g}$',
               transform=ax[0].transAxes, va='top', ha='left', fontsize=12, bbox=_bb)

    edges = np.linspace(U.min(), U.max(), 41)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    hMg = np.histogram(U, bins=edges, weights=dMg)[0]
    hC4 = np.histogram(U, bins=edges, weights=dC4)[0]
    ax[1].step(ctr, hMg / hMg.max(), where='mid', color='orange', lw=3,
               label=r'$dL_{\rm MgII}/d\log U$')
    ax[1].step(ctr, hC4 / hC4.max(), where='mid', color='purple', lw=3,
               label=r'$dL_{\rm CIV}/d\log U$')
    ax[1].axvspan(-1.5, -0.5, color='cyan', alpha=0.3)
    ax[1].set_xlabel(r'$\log U$')
    ax[1].set_ylabel(r'LOC-weighted emission')
    ax[1].legend(fontsize=12)
    for a in ax:
        a.minorticks_on()
        a.tick_params(top=True, right=True, which='major', length=8, width=1.5, direction='in')
        a.tick_params(top=True, right=True, which='minor', length=4, width=1, direction='in')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
