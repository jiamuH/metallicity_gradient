"""The 1650-1840 A region of the Constantin & Shields (2003) NLS1 average
composite (recovered from their Figure 4), with their fitted Lorentzian lines
(Table 3: centre, FWHM, equivalent width) and a straight-line continuum fitted
underneath.  Checks whether their N III] 1750 deblend reproduces the spectrum.

Run (from the project root): python3 nitrogen/plot_cs03_n3_region.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy', 'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

DATA = 'data/nitrogen/cs03_nls1_average_composite.txt'
OUT = 'plots/nitrogen/cs03_n3_region.png'
C_KMS = 2.998e5
WMIN, WMAX = 1650.0, 1840.0

# Constantin & Shields (2003) Table 3: label, fitted centre [A], FWHM [km/s], EW [A]
LINES = [
    (r'$\rm O\,III]$',            1665.4, 2500, 6.2),
    (r'$\rm Al\,II$',             1680.0, 2250, 1.9),
    (r'$\rm N\,IV+Al\,II$',       1725.6, 4710, 3.3),
    (r'$\rm N\,III]$',            1750.5, 1480, 3.0),
    (r'$\rm Fe\,II~UV191$',       1785.3, 1430, 1.8),
    (r'$\rm Si\,II$',             1816.9, 1250, 0.9),
]
COLORS = ['#999999', '#999999', '#0072B2', '#D55E00', '#009E73', '#999999']


def lorentz(w, w0, fwhm_kms, ew):
    """Lorentzian normalised so that its integral equals ew (in A)."""
    g = 0.5 * fwhm_kms / C_KMS * w0            # half width at half maximum [A]
    return ew * g / np.pi / ((w - w0) ** 2 + g ** 2)


def main():
    w, f = np.loadtxt(DATA, unpack=True)
    s = (w > WMIN) & (w < WMAX)
    w, f = w[s], f[s]
    comps = [lorentz(w, c, fw, ew) for _, c, fw, ew in LINES]
    shape = 1.0 + np.sum(comps, axis=0)          # f = continuum * (1 + sum of lines / continuum)
    a, b = np.polyfit(w, f / shape, 1)
    cont = a * w + b
    model = cont * shape

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
    ax.plot(w, f, color='k', lw=1.2, drawstyle='steps-mid',
            label=r'$\rm composite~(recovered)$')
    ax.plot(w, model, color='#CC79A7', lw=2.5, label=r'$\rm Constantin~\&~Shields~fit$')
    ax.plot(w, cont, color='0.5', ls='--', lw=1.5)
    for (lab, c, _, _), comp, col in zip(LINES, comps, COLORS):
        ax.plot(w, cont * (1 + comp), color=col, lw=1.8)
        ax.text(c, cont[np.argmin(abs(w - c))] * (1 + comp.max()) + 0.04, lab,
                fontsize=13, ha='center', va='bottom', color=col)
    axr.plot(w, f - model, color='k', lw=1.2, drawstyle='steps-mid')
    axr.axhline(0, color='0.5', ls='--', lw=1.2)
    ax.set_ylabel(r'$F_\lambda~[\rm normalised]$')
    axr.set_ylabel(r'$\rm data-fit$')
    axr.set_xlabel(r'$\rm rest\mbox{-}frame~wavelength~[\AA]$')
    ax.legend(loc='upper right', fontsize=14, frameon=False)
    ax.set_ylim(0.85, 1.75)
    for a_ in (ax, axr):
        a_.minorticks_on()
        a_.tick_params(axis='both', which='major', direction='in', top=True, right=True,
                       length=8, width=1.5)
        a_.tick_params(axis='both', which='minor', direction='in', top=True, right=True,
                       length=4, width=1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close(fig)
    rms = np.std(f - model)
    print(f'continuum slope {a:.2e} /A, residual rms {rms:.3f}; Saved {OUT}')


if __name__ == '__main__':
    main()
