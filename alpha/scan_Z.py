"""Scan metallicity: LOC-averaged MgII/CIV vs Z_norm (k=0, p=0.35).
Does the MgII/CIV excess go away at lower Z?

    python3 -m alpha.scan_Z
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from alpha.loc_gradient_fit import make_interpolators, model_line_ratios, OBS_RANGE, Z_GRID

K, P, LOGQ = 0.0, 0.35, 55.0
R_REFS = [16.7, 17.2, 17.5, 17.9]
OUT = 'plots/alpha/loc_gradient_noA/scan_Z.png'


def main():
    interps = make_interpolators()
    obs = OBS_RANGE['MgII/CIV'][0]
    Z = np.logspace(np.log10(Z_GRID.min()), np.log10(Z_GRID.max()), 50)
    print(f"observed log(MgII/CIV) = {obs:+.3f};  Z grid = [{Z_GRID.min():g}, {Z_GRID.max():g}]")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    cmap = plt.colormaps['viridis']
    for i, lrr in enumerate(R_REFS):
        vals = np.array([np.log10(model_line_ratios(interps, LOGQ, K, P,
                         log_r_ref=lrr, z_norm=z)[0]['MgII/CIV'][0]) for z in Z])
        c = cmap(i / (len(R_REFS) - 1))
        ax.plot(Z, vals, '-', lw=3, color=c, label=rf'$\log r_{{\rm ref}}={lrr:g}$')
        s = vals - obs
        idx = np.where(np.diff(np.sign(s)))[0]
        if idx.size:
            j = idx[0]
            zc = 10 ** (np.log10(Z[j]) + (np.log10(Z[j + 1]) - np.log10(Z[j]))
                        * (-s[j]) / (s[j + 1] - s[j]))
            ax.plot(zc, obs, 'o', color=c, ms=12, mec='k', mew=1.2, zorder=5)
            print(f"  log_r_ref={lrr:g}:  Z matching observed = {zc:.2f} Z_sun")
        else:
            print(f"  log_r_ref={lrr:g}:  no crossing (model {'>' if vals.min() > obs else '<'} observed over full Z range)")
    ax.axhline(obs, color='c', ls='--', lw=2.5, label='SDSS-RM mean')
    ax.axvline(3.0, color='0.5', ls=':', lw=2, label=r'current $Z=3\,Z_\odot$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$Z_{\rm norm}~[Z_\odot]$')
    ax.set_ylabel(r'LOC-averaged $\log(\rm Mg\,II\,/\,C\,IV)$')
    ax.legend(fontsize=11, loc='best')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, which='major', length=8, width=1.5, direction='in')
    ax.tick_params(top=True, right=True, which='minor', length=4, width=1, direction='in')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == '__main__':
    main()
