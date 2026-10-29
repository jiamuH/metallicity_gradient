"""Alpha-element gradient slope k of the 71 well-constrained SDSS-RM fits
against black-hole mass, coloured by Eddington ratio, with the NLS1 mass range
of Section 2 marked along the bottom axis (reply to referee comment 4, Figure R1).

Run: python3 -m alpha.plot_k_vs_mbh   (from the project root)
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from alpha.sdssrm_sample_properties import sample
from nitrogen.csh_nls1_properties import compute

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy', 'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

OUT = 'plots/sample_comparison/k_vs_mbh.png'


def main():
    df = sample().dropna(subset=['logBH'])
    nls1_m = np.array([r['log_m'] for r in compute()])
    fig, ax = plt.subplots(figsize=(9, 7))
    # one tick per NLS1 mass along the bottom of the panel
    ax.vlines(nls1_m, -1.0, -0.9, color='#D55E00', lw=2.5, zorder=4)
    ax.text(5.7, -0.86, r'$\rm NLS1~masses$', fontsize=15, color='#A04000',
            ha='left', va='bottom')
    ax.axhline(np.median(df.k), color='0.4', ls='--', lw=1.5, zorder=1)
    ax.axhline(0, color='0.7', ls=':', lw=1.2, zorder=1)
    norm = Normalize(df.logEdd.min(), df.logEdd.max())
    ax.errorbar(df.logBH, df.k, yerr=[df.k_lo, df.k_hi], fmt='none', ecolor='0.6',
                elinewidth=1, capsize=0, zorder=2)
    sc = ax.scatter(df.logBH, df.k, c=df.logEdd, cmap='viridis', norm=norm, s=60,
                    edgecolor='white', lw=0.8, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r'$\log\,L_{\rm bol}/L_{\rm Edd}$')
    cbar.ax.tick_params(direction='in', length=6, width=1.5)
    ax.set_xlim(5.6, 10.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel(r'$\log\,M_{\rm BH}~[M_\odot]$')
    ax.set_ylabel(r'$k$')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True,
                   length=8, width=1.5)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True,
                   length=4, width=1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
