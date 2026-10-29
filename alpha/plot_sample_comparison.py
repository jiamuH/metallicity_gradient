"""Compare the 21 Constantin & Shields (2003) NLS1s (Section 2) with the 207
SDSS-RM quasars fitted in Section 3.

Two candidate figures for referee comment 3:
  sample_comparison_mass_lum.png    black-hole mass vs bolometric luminosity,
                                    with lines of constant Eddington ratio
  sample_comparison_histograms.png  redshift, luminosity, mass and Eddington
                                    ratio histograms

Run: python3 -m alpha.plot_sample_comparison   (from the project root)
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from nitrogen.csh_nls1_properties import compute, BC_5100
from alpha.sdssrm_sample_properties import sample

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy', 'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

OUT_DIR = 'plots/sample_comparison'
NLS1_COLOR, RM_COLOR = '#D55E00', '#0072B2'
NLS1_LABEL = r'$\rm NLS1s~(Constantin~\&~Shields~2003)$'
RM_LABEL = r'$\rm SDSS\mbox{-}RM~quasars$'
LOG_EDD_CONST = np.log10(1.26e38)
SIG_LOG_M, SIG_LOG_LBOL = 0.4, 0.1   # dex, see plot_mass_lum


def load():
    nls1 = compute()
    nls1 = {'z': np.array([r['z'] for r in nls1]),
            'log_lbol': np.array([np.log10(BC_5100) + r['log_l5100'] for r in nls1]),
            'log_m': np.array([r['log_m'] for r in nls1]),
            'log_edd': np.array([r['log_edd'] for r in nls1])}
    df = sample(well_constrained=False)
    rm = {'z': df['zsys'].values, 'log_lbol': df['logLBol'].values,
          'log_m': df['logBH'].values, 'log_edd': df['logEdd'].values}
    return nls1, rm


def _ticks(ax, major=8, minor=4):
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True,
                   length=major, width=1.5)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True,
                   length=minor, width=1)


def plot_mass_lum(nls1, rm, out):
    fig, ax = plt.subplots(figsize=(8, 7))
    m = np.linspace(5.5, 10.0, 2)
    for log_edd in (0, -1, -2):
        ax.plot(m, m + LOG_EDD_CONST + log_edd, color='0.6', lw=1.5, ls='--', zorder=0)
        # label sits on its line, which is broken by the white label box
        x = {0: 5.9, -1: 6.4, -2: 7.6}[log_edd]
        ax.text(x, x + LOG_EDD_CONST + log_edd,
                r'$\lambda_{\rm Edd}=%s$' % {0: '1', -1: '0.1', -2: '0.01'}[log_edd],
                fontsize=14, color='0.4', rotation=45, transform_rotates_text=True,
                rotation_mode='anchor', ha='center', va='center', zorder=1,
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    ax.scatter(rm['log_m'], rm['log_lbol'], s=60, marker='o', color=RM_COLOR,
               edgecolor='white', lw=0.8, label=RM_LABEL, zorder=2)
    ax.scatter(nls1['log_m'], nls1['log_lbol'], s=80, marker='s', color=NLS1_COLOR,
               edgecolor='white', lw=0.8, label=NLS1_LABEL, zorder=3)
    # typical systematic uncertainty: single-epoch mass scatter (0.43 dex H-beta,
    # 0.36 dex C IV; Vestergaard & Peterson 2006) and bolometric-correction
    # scatter (10.4 +- 2.5, ~0.1 dex; Richards et al. 2006)
    ax.errorbar(9.0, 43.1, xerr=SIG_LOG_M, yerr=SIG_LOG_LBOL, fmt='none', ecolor='0.3',
                elinewidth=2, capsize=4, capthick=2)
    ax.text(9.0, 43.1 + SIG_LOG_LBOL + 0.08, r'$\rm typical~uncertainty$', fontsize=13,
            color='0.3', ha='center', va='bottom')
    ax.set_xlim(5.5, 10.0)
    ax.set_ylim(42.5, 47.5)
    ax.set_xlabel(r'$\log\,M_{\rm BH}~[M_\odot]$')
    ax.set_ylabel(r'$\log\,L_{\rm bol}~[\rm erg~s^{-1}]$')
    ax.legend(loc='upper left', fontsize=14, frameon=False)
    _ticks(ax)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def plot_histograms(nls1, rm, out):
    panels = [('z', r'$z$', 0.1),
              ('log_lbol', r'$\log\,L_{\rm bol}~[\rm erg~s^{-1}]$', 0.25),
              ('log_m', r'$\log\,M_{\rm BH}~[M_\odot]$', 0.25),
              ('log_edd', r'$\log\,L_{\rm bol}/L_{\rm Edd}$', 0.25)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, (key, label, width) in zip(axes.flat, panels):
        vals = np.concatenate([nls1[key], rm[key][np.isfinite(rm[key])]])
        bins = np.arange(np.floor(vals.min() / width) * width, vals.max() + width, width)
        ax.hist(rm[key][np.isfinite(rm[key])], bins=bins, color=RM_COLOR, alpha=0.6,
                label=RM_LABEL)
        ax.hist(nls1[key], bins=bins, histtype='step', color=NLS1_COLOR, lw=3,
                label=NLS1_LABEL)
        ax.set_xlabel(label)
        ax.set_ylabel(r'$\rm Number$')
        _ticks(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3,
               fontsize=14, frameon=False)
    fig.subplots_adjust(hspace=0.3, wspace=0.25)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nls1, rm = load()
    plot_mass_lum(nls1, rm, f'{OUT_DIR}/sample_comparison_mass_lum.png')
    plot_histograms(nls1, rm, f'{OUT_DIR}/sample_comparison_histograms.png')


if __name__ == '__main__':
    main()
