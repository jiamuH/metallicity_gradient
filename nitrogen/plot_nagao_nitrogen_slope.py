"""Nitrogen abundance from N V and N IV] (ratios to C IV) against LOC radius
normalised to the N V radius (so no radius--luminosity scaling is needed),
for the Constantin & Shields (2003) NLS1 composite and the Nagao et al. (2006a)
SDSS quasar composites at 2.0 <= z < 2.5 (reply to referee comment 4).

Run (from the project root): python3 nitrogen/plot_nagao_nitrogen_slope.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from loc_nitrogen_abundance import CSH, consistent_loc_radii   # noqa: E402
from _nagao_nitrogen_slope import nagao_table3, two_point     # noqa: E402

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy', 'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

OUT = 'plots/nitrogen/nagao_nitrogen_two_point.png'
MB_LABELS = [r'$-24.5>M_B\geq-25.5$', r'$-25.5>M_B\geq-26.5$',
             r'$-26.5>M_B\geq-27.5$', r'$-27.5>M_B\geq-28.5$']


def main():
    radii = consistent_loc_radii()
    r = np.array([1.0, radii['N4'][0] / radii['N5'][0]])   # r / r(N V)
    nagao = [d for d in nagao_table3().values() if 'N4' in d]
    blues = plt.cm.Blues(np.linspace(0.45, 0.95, len(nagao)))
    sets = [(r'$\rm NLS1s~(Constantin~\&~Shields~2003)$', CSH, '#D55E00', 's', 0.0)]
    sets += [(r'$\rm Nagao~et~al.~(2006a),~$' + lab, d, c, 'o', 0.006 * (i + 1))
             for i, (lab, d, c) in enumerate(zip(MB_LABELS, nagao, blues))]

    fig, ax = plt.subplots(figsize=(9, 7))
    for label, src, color, marker, dx in sets:
        res, slope = two_point(src, radii, 'C4')
        rr = r * 10 ** dx          # small horizontal offsets so points do not overlap
        x = np.array([res['N5'][0], res['N4'][0]])
        ax.plot(rr, x, '-', color=color, lw=2, zorder=2)
        for k, ion in enumerate(('N5', 'N4')):
            v, lo, hi, _ = res[ion]
            ax.errorbar(rr[k], v, yerr=[[v - lo], [hi - v]], fmt=marker, ms=11,
                        mfc=color, mec=color, mew=2,
                        ecolor=color, elinewidth=2, capsize=4, zorder=3,
                        label=label if k == 0 else None)
    ax.axhline(1.0, color='0.6', ls=':', lw=1.5, zorder=1)
    for ion, name in (('N5', r'$\rm N\,V$'), ('N4', r'$\rm N\,IV]$')):
        ax.text({'N5': 1.0, 'N4': r[1]}[ion] * 1.03, 0.13, name, fontsize=16, ha='center', color='0.3')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.75, 3.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([r'$1$', r'$2$', r'$3$'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(0.1, 40)
    ax.set_xlabel(r'$r/r_{\rm N\,V}$')
    ax.set_ylabel(r'$\rm (N/O)/(N/O)_{\odot}$')
    ax.legend(loc='upper right', fontsize=12, frameon=False)
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
