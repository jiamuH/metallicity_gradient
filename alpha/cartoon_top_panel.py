#!/usr/bin/env python3
"""Top panel of the enrichment cartoon (Fig. 7): model N and C+N+O abundance
profiles vs distance from the SMBH.

Design (agreed 2026-08-18):
- C+N+O (blue): nearly flat ~3-4x solar across the BLR (16-84% band from the
  population fits), declining outward to the NLR mean (~1.26, i.e. +0.1 dex).
- N (orange): metamorphic floor at the solar-pattern share of the total
  (constant N/(C+N+O) ~ 1.3x solar ratio) plus an immortal-star excess
  rising inward as ~r^-1.  N stays slightly above C+N+O everywhere.
- Solid where Fig. 6 has data (~10^2-10^4 r_g for M_BH = 1e8 Msun,
  1 ld ~ 175 r_g); dashed beyond (torus + NLR, model extrapolation).

Output: plots/alpha/cartoon_top_panel.png  (standalone review version with
x-axis labels; the stitched version will hide them).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
    'font.family': 'serif', 'font.weight': 'heavy', 'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

N_COLOR = '#E8471C'
CNO_COLOR = '#3B6FE3'

X_SOLID_MAX = 4.1          # log10 r_g outer edge of the measured range
X_TORUS = 5.0              # log10 r_g inner edge of the dusty torus (drawn art)
RATIO_FLOOR = 1.3          # (N/(C+N+O)) / solar ratio of the metamorphic yield

# C+N+O relative to solar: control points in (log10 r_g, log10 T)
_T_PTS_X = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
_T_PTS_Y = np.log10([5.0, 3.6, 2.6, 2.0, 1.5, 1.26, 1.22])
_logT = PchipInterpolator(_T_PTS_X, _T_PTS_Y)


def T(x):
    return 10 ** _logT(x)


def N(x):
    floor = RATIO_FLOOR * T(x)
    excess = 8.5 * 10 ** (-(np.asarray(x) - 2.0))   # ~r^-1 immortal-star term
    return floor + excess


def main(out='plots/alpha/cartoon_top_panel.png', show_xlabels=True):
    fig, ax = plt.subplots(figsize=(12.5, 3.4))
    xs = np.linspace(2.0, X_SOLID_MAX, 200)
    xd = np.linspace(X_SOLID_MAX, 8.0, 300)

    # 16-84% band on the C+N+O curve over the measured range (+/-0.25 dex)
    ax.fill_between(10 ** xs, T(xs) * 10 ** -0.25, T(xs) * 10 ** 0.25,
                    color=CNO_COLOR, alpha=0.18, lw=0, zorder=1)

    ax.plot(10 ** xs, T(xs), '-', color=CNO_COLOR, lw=3.2, zorder=3,
            label=r'$\rm C+N+O$')
    ax.plot(10 ** xd, T(xd), '--', color=CNO_COLOR, lw=3.2, zorder=3)
    ax.plot(10 ** xs, N(xs), '-', color=N_COLOR, lw=3.2, zorder=4,
            label=r'$\rm N$')
    ax.plot(10 ** xd, N(xd), '--', color=N_COLOR, lw=3.2, zorder=4)

    ax.axvline(10 ** X_TORUS, color='grey', ls='--', lw=1.5, zorder=2)

    ax.text(0.20, 0.86, r'$\rm immortal~stars$' + '\n' + r'$\rm enrich~N$',
            transform=ax.transAxes, ha='center', va='top', fontsize=15, color='0.25')
    ax.text(0.71, 0.62,
            r'$\rm enrichment~by~metamorphic~stars$' + '\n'
            + r'$\rm N~\&~C+N+O~rise~together$',
            transform=ax.transAxes, ha='center', va='top', fontsize=15, color='0.25')

    ax.set_xscale('log')
    ax.set_xlim(1e2, 1e8)
    ax.set_ylim(0, 16)
    ax.set_ylabel(r'$\rm Abundance~Rel.~Solar$', fontsize=18)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True,
                   length=8, width=1.5, direction='in', labelsize=16)
    ax.tick_params(axis='both', which='minor', top=True, right=True,
                   length=4, width=1, direction='in')
    if show_xlabels:
        ax.set_xlabel(r'$\rm Distance~from~SMBH~[\it{r_g}\rm ]$', fontsize=18)
    else:
        ax.tick_params(axis='x', labelbottom=False)
    ax.legend(loc='upper right', fontsize=15, frameon=False)

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
