"""Scan the LOC radial-cover exponent Gamma: where does the LOC-averaged
MgII/CIV match the SDSS-RM observed mean?

    python3 -m alpha.scan_gamma
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import alpha.loc_gradient_fit as M
from alpha.loc_gradient_fit import make_interpolators, model_line_ratios, OBS_RANGE

K, P, Z, LOGQ = 0.0, 0.35, 3.0, 55.0
R_REFS = [16.7, 17.2, 17.5, 17.9]
GAMMAS = np.linspace(-5.0, 1.0, 61)
OUT = 'plots/alpha/loc_gradient_noA/scan_gamma.png'


def main():
    interps = make_interpolators()
    obs = OBS_RANGE['MgII/CIV'][0]
    g0 = M.GAMMA_LOC
    print(f"observed log(MgII/CIV) = {obs:+.3f};  current Gamma = {g0:g}")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cmap = plt.colormaps['viridis']
    for i, lrr in enumerate(R_REFS):
        vals = []
        for g in GAMMAS:
            M.GAMMA_LOC = g
            rr, _, _ = model_line_ratios(interps, LOGQ, K, P, log_r_ref=lrr, z_norm=Z)
            vals.append(np.log10(rr['MgII/CIV'][0]))
        vals = np.array(vals)
        c = cmap(i / (len(R_REFS) - 1))
        ax.plot(GAMMAS, vals, '-', lw=3, color=c, label=rf'$\log r_{{\rm ref}}={lrr:g}$')
        s = vals - obs
        idx = np.where(np.diff(np.sign(s)))[0]
        cross = np.nan
        if idx.size:
            j = idx[0]
            cross = GAMMAS[j] + (GAMMAS[j + 1] - GAMMAS[j]) * (-s[j]) / (s[j + 1] - s[j])
            ax.plot(cross, obs, 'o', color=c, ms=12, mec='k', mew=1.2, zorder=5)
        print(f"  log_r_ref={lrr:g}:  Gamma matching observed = {cross:+.2f}")
    M.GAMMA_LOC = g0
    ax.axhline(obs, color='c', ls='--', lw=2.5, label='SDSS-RM mean')
    ax.axvline(g0, color='0.5', ls=':', lw=2, label=rf'current $\Gamma={g0:g}$')
    ax.set_xlabel(r'$\Gamma$ (LOC radial-cover exponent)')
    ax.set_ylabel(r'LOC-averaged $\log(\rm Mg\,II\,/\,C\,IV)$')
    ax.legend(fontsize=11, loc='upper left')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, which='major', length=8, width=1.5, direction='in')
    ax.tick_params(top=True, right=True, which='minor', length=4, width=1, direction='in')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
