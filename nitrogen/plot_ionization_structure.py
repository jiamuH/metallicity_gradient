#!/usr/bin/env python3
"""Nitrogen ionization structure of a BLR cloud at Z_alpha = 1 vs 10 Z_sun.

Appendix figure: explains why N III]/C IV is sensitive to the overall
(C+N+O) abundance while N IV]/C IV and N V/C IV are not.  Raising the
metallicity tenfold cools the cloud and collapses the high-ionization
skin, so the C IV / N IV] / N V emitting gas shrinks while the
partially ionized N III] zone grows.

Top panel  : Z_alpha = 1  Z_sun
Bottom panel: Z_alpha = 10 Z_sun
Both panels share the depth axis.

Data sources (one solar-nitrogen model per metallicity):
    _nitrogen.ovr  -- 'save element nitrogen': N2+, N3+, N4+ vs depth
    _blr.ovr       -- 'save overview':         C3+ (C IV), Te, H+ vs depth

These come from the single-model decks nagao_ionstruct/ionstruct_m{1,10}.in.

Run from repo root:
    python3 nitrogen/plot_ionization_structure.py
"""
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# Nitrogen palette, matched to plot_n3n4n5_line_ratios.py so each ion is
# colored as the nitrogen line it produces.
_CUSTOM_COLORS = [
    (0.15, 0.08, 0.40), (0.294, 0.161, 0.569), (0.529, 0.173, 0.635),
    (0.753, 0.212, 0.616), (0.918, 0.310, 0.533), (0.980, 0.471, 0.463),
    (0.965, 0.663, 0.478), (0.929, 0.851, 0.639),
]
CUSTOM_CMAP = LinearSegmentedColormap.from_list('cc', _CUSTOM_COLORS, N=256)
COL_N3, COL_N4, COL_N5 = [CUSTOM_CMAP(x) for x in np.linspace(0.1, 0.9, 3)]
CIV_COLOR = 'orangered'   # C3+ reference curve (distinct from the N ions)
TE_GREY = '0.45'          # T_e curve and its axis

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy',
                     'font.size': 20})
plt.rcParams['text.latex.preamble'] = (
    r'\usepackage{amsmath} \usepackage{bm} \boldmath')

BASE = Path('/Users/jiamuh/c23.01/my_models/nagao_ionstruct')
RUN = {
    1:  (BASE, 'ionstruct_m1'),
    10: (BASE, 'ionstruct_m10'),
}
GRID_IDX = 0   # single-model decks: one block per file
OUT = Path('plots/nitrogen/ionization_structure.png')

# _nitrogen.ovr columns: depth N0 N+ N2+ N3+ N4+ N5+ N6+ N7+
NIT = dict(depth=0, N2p=3, N3p=4, N4p=5)
# _blr.ovr (overview) columns
OVR = dict(depth=0, Te=1, HII=7, C3p=15)

# (column key, color, legend label)
N_IONS = [('N2p', COL_N3, r'$\rm N^{2+}~(N\,III])$'),
          ('N3p', COL_N4, r'$\rm N^{3+}~(N\,IV])$'),
          ('N4p', COL_N5, r'$\rm N^{4+}~(N\,V)$')]


def block(path, idx):
    """Return the idx-th grid point of a Cloudy grid 'save' file."""
    out, cur = [], []
    with open(path) as f:
        for line in f:
            if 'GRID_DELIMIT' in line:
                if cur:
                    out.append(np.array(cur))
                cur = []
                continue
            if line.lstrip().startswith('#'):
                continue
            p = line.split()
            if len(p) >= 8:
                try:
                    cur.append([float(x) for x in p])
                except ValueError:
                    pass
    if cur:
        out.append(np.array(cur))
    return out[idx]


def main():
    # check the re-run output exists
    missing = [str(d / f'{pre}_nitrogen.ovr')
               for d, pre in RUN.values()
               if not (d / f'{pre}_nitrogen.ovr').exists()]
    if missing:
        sys.exit('Missing nitrogen save files:\n  ' + '\n  '.join(missing)
                 + '\nRun the decks nagao_ionstruct/ionstruct_m{1,10}.in first.')

    data = {}
    for z, (d, pre) in RUN.items():
        nit = block(d / f'{pre}_nitrogen.ovr', GRID_IDX)
        ovr = block(d / f'{pre}_blr.ovr', GRID_IDX)
        data[z] = (nit, ovr)

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.6, 11))

    for ax, z in zip(axes, RUN):
        nit, ovr = data[z]
        dN, dO = nit[:, NIT['depth']], ovr[:, OVR['depth']]

        for key, color, _ in N_IONS:
            ax.plot(dN, nit[:, NIT[key]], color=color, lw=3.6)
        ax.plot(dO, ovr[:, OVR['C3p']], color=CIV_COLOR, lw=3.6, ls='-.')  # C IV

        cross = np.where(ovr[:, OVR['HII']] < 0.5)[0]
        if cross.size:
            ax.axvline(dO[cross[0]], color='0.5', ls=':', lw=2.2)

        axR = ax.twinx()
        axR.plot(dO, ovr[:, OVR['Te']] / 1e4, color=TE_GREY, ls='--', lw=2.6)
        axR.set_ylim(0, 2.4)
        axR.minorticks_on()
        axR.tick_params(which='major', direction='in', right=True,
                        length=9, width=1.5, colors=TE_GREY)
        axR.tick_params(which='minor', direction='in', right=True,
                        length=4.5, width=1.0, colors=TE_GREY)
        axR.set_ylabel(r'$T_e~[10^4~\rm K]$', color=TE_GREY)
        axR.spines['right'].set_color(TE_GREY)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e8, 1.5e12)
        ax.set_ylim(1e-5, 2.0)
        ax.set_ylabel(r'$\rm Ionization~fraction$')
        ax.minorticks_on()
        ax.tick_params(which='major', direction='in', top=True, left=True,
                       length=9, width=1.5)
        ax.tick_params(which='minor', direction='in', top=True, left=True,
                       length=4.5, width=1.0)
        ax.text(0.05, 0.975, rf'$Z_{{\alpha}}={z}\,Z_{{\odot}}$',
                transform=ax.transAxes, ha='left', va='top', fontsize=23)

    axes[-1].set_xlabel(r'$\rm Depth~[cm]$')

    handles = [Line2D([], [], color=c, lw=3.6) for _, c, _ in N_IONS]
    handles += [Line2D([], [], color=CIV_COLOR, lw=3.6, ls='-.'),
                Line2D([], [], color=TE_GREY, ls='--', lw=2.6),
                Line2D([], [], color='0.5', ls=':', lw=2.2)]
    labels = [lbl for _, _, lbl in N_IONS]
    labels += [r'$\rm C^{3+}~(C\,IV)$', r'$T_e$', r'$\rm H^{+}~front$']
    axes[0].legend(handles, labels, fontsize=12.5, loc='lower left',
                   frameon=False, handlelength=2.4, ncol=2,
                   columnspacing=1.1)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    print(f'Saved: {OUT}')

    for z in RUN:
        nit, ovr = data[z]
        dN, dO = nit[:, NIT['depth']], ovr[:, OVR['depth']]
        n2 = dN[nit[:, NIT['N2p']] >= 0.05]
        n4 = dN[nit[:, NIT['N4p']] >= 0.05]
        ifr = dO[ovr[:, OVR['HII']] < 0.5]
        print(f'  Z_alpha={z:2d}: N2+>=0.05 zone {n2.min():.1e}-{n2.max():.1e} cm'
              f' | N4+>=0.05 skin to {(n4.max() if n4.size else 0):.1e} cm'
              f' | H+ front {(ifr[0] if ifr.size else np.nan):.1e} cm'
              f' | Te {ovr[:,OVR["Te"]].min():.0f}-{ovr[:,OVR["Te"]].max():.0f} K')
    plt.close(fig)


if __name__ == '__main__':
    main()
