"""How strongly do LOC-integrated broad-line ratios respond to a uniform
metallicity (k = 0) from 0.01 to 100 Z_sun?  Uses the standard Cloudy grid plus
the low-Z and high-Z extension runs.  Each ratio is shown relative to its value
at Z_sun.  Lines use the median fitted log r_ref; the shaded band spans log r_ref
at the 16th to 84th percentiles of the fits (median p and median log Q of the
sample).  The grey band is the 0.05-20 Z_sun range used in the fits.

Run: python3 -m alpha.plot_z_saturation_extended
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import alpha.loc_gradient_fit as lgf
from alpha._check_z_saturation_extended import extended_interpolator, model_at, FIT_CSV
from alpha.plot_Z_profiles import _logQ_per_object

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy', 'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

OUT = 'plots/alpha/z_saturation_extended.png'
# label, lines for the interpolator, which model_ratios_at output to use (0: L0/L1, 1: (L2+L3)/L1), color
RATIOS = [
    (r'$\rm Mg\,II/C\,IV$',            ('Mg2', 'C4', 'Si4', 'O4'), 0, '#0072B2'),
    (r'$\rm (Si\,IV+O\,IV])/C\,IV$',   ('Mg2', 'C4', 'Si4', 'O4'), 1, '#D55E00'),
    (r'$\rm C\,IV/He\,II$',            ('C4', 'He2', 'C4', 'C4'), 0, '#009E73'),
    (r'$\rm N\,V/C\,IV$',              ('N5', 'C4', 'C4', 'C4'), 0, '#CC79A7'),
    (r'$\rm N\,V/He\,II$',             ('N5', 'He2', 'C4', 'C4'), 0, '#E69F00'),
]


def main():
    std_lo, std_hi = lgf.Z_GRID.min(), lgf.Z_GRID.max()
    fits = pd.read_csv(FIT_CSV).dropna(subset=['k'])
    logq = np.median(_logQ_per_object(fits.rm_id.tolist()).values)
    p = fits.p.median()
    lrr = np.percentile(fits.log_r_ref, [16, 50, 84])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axvspan(np.log10(std_lo), np.log10(std_hi), color='0.92', zorder=0)
    for label, lines, which, color in RATIOS:
        lgf.Z_GRID = np.concatenate([lgf._Z_SUBSOLAR, lgf._Z_SUPER])   # standard grid before reloading
        interp, z = extended_interpolator(lines)
        lgf.Z_GRID = z      # widen the clipping range in model_ratios_at to the extended grid
        i_sun = np.argmin(abs(z - 1.0))
        curves = []
        for lr in lrr:
            r = np.array([model_at(interp, logq, p, lr, zz)[which] for zz in z])
            curves.append(np.log10(r) - np.log10(r[i_sun]))
        lo, mid, hi = np.min(curves, axis=0), curves[1], np.max(curves, axis=0)
        ext = (z < std_lo) | (z > std_hi)
        ax.fill_between(np.log10(z), lo, hi, color=color, alpha=0.2, lw=0)
        ax.plot(np.log10(z), mid, color=color, lw=2.2, label=label)
        ax.plot(np.log10(z[ext]), mid[ext], 'o', color=color, ms=7, mec='white', mew=1.2)
    ax.text(0.5 * (np.log10(std_lo) + np.log10(std_hi)), 0.97, r'$\rm fitted~grid$',
            transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=15, color='0.4')
    ax.set_xlabel(r'$\log Z~[Z_\odot]$')
    ax.set_ylabel(r'$\log~({\rm ratio}) - \log~({\rm ratio~at}~Z_\odot)$')
    ax.set_xlim(-2.1, 2.1)
    ax.legend(loc='upper left', fontsize=14, frameon=False)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=8, width=1.5)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=4, width=1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
