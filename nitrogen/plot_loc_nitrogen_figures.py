#!/usr/bin/env python3
"""LOC-averaged versions of the nitrogen abundance figures (Figs. 3 and 6).

Uses the LOC-integrated line ratios from loc_nitrogen_abundance.py instead of the
single-zone models behind the published figures, and writes everything to new
files so the existing results stay retrievable:

    nitrogen/nitrogen_abundance_vs_r_loc.dat
    plots/nitrogen/Nitrogen_abundance_vs_rBLR_loc.png            (Fig. 3 analogue)
    plots/alpha/loc_gradient_noA_Qref58/Z_profiles_population_loc.png  (Fig. 6)

Primary inference: the metals-3 LOC grid (N23_m3), Van den Berk 2001 composite,
C IV denominator, which is the direct LOC counterpart of the published analysis.
Because that grid holds nitrogen fixed, X is obtained by linear scaling from the
solar-N model; the metals-1 grids show the ratios go as X^0.9, so this is good to
about 10% in X.  N/H = X * Z_alpha with Z_alpha = 3.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from compute_nitrogen_abundance import fit_powerlaw_mcmc
from loc_nitrogen_abundance import (GRIDS, load_grid, model_ratio_curve,
                                    obs_ratio, invert, VDB, CSH, IONS)

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy',
                     'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

PRIMARY = dict(grid='N23_m3', src=VDB, sname='VdB01', den='C4')

ION_COLOR = {'N3': '#5B2B7B', 'N4': '#C2185B', 'N5': '#E67E22'}
ION_LABEL = {'N3': r'$\rm N\,III]$', 'N4': r'$\rm N\,IV]$', 'N5': r'$\rm N\,V$'}
ION_POT = {'N3': 47.4, 'N4': 77.5, 'N5': 97.9}

RADII_DAT = Path('nitrogen/nitrogen_abundance_vs_r.dat')
OUT_DAT = Path('nitrogen/nitrogen_abundance_vs_r_loc.dat')


def loc_radii():
    """{ion: (r, r_lo, r_hi)} in light-days, on the abundance-consistent cloud
    ensemble (same phi/n range and matter-bounded cut as the LOC inference).
    The published nitrogen_abundance_vs_r.dat radii used the full stitched grid
    (phi 16-23, n 7-12, no cut) and place the outer ions ~0.2 dex farther out."""
    from loc_nitrogen_abundance import consistent_loc_radii
    return consistent_loc_radii()


def infer_loc(src, den):
    """{ion: (X, X_lo, X_hi, off_grid)} at Z_alpha = 3.

    Uses the metals-3 normalisation with the metals-1 R(X) shape (see
    plot_loc_nitrogen_composites.curve_at_z3), so all LOC figures share one
    inference method."""
    from plot_loc_nitrogen_composites import infer
    out = {}
    for ion in IONS:
        if ion not in src or den not in src:
            continue
        out[ion] = infer(ion, src, den)
    return out


Z_ADOPT = 3.0


def write_dat(res, radii, path=OUT_DAT):
    """Write in the schema alpha/plot_Z_profiles.py:_load_nitrogen_points expects."""
    lines = ['# LOC-averaged nitrogen abundance (N/H)/(N/H)_sun vs BLR radius',
             '# LOC integral over the (phi, n_H) grid, Gamma = beta = -1,',
             '# full grid on the common window (no cut); metals-3 normalisation with',
             '# metals-1 R(X) shape; N/H = X * Z_alpha at Z_alpha = 3',
             '# Columns: species, r_blr [ld], r_low, r_high, model, Z, Z_low, Z_high']
    for ion, (x, lo, hi, off) in res.items():
        r, r_lo, r_hi = radii[ion]
        lines.append(f'{ion}  {r:.6f}  {r_lo:.6f}  {r_hi:.6f}  loc  '
                     f'{x * Z_ADOPT:.6f}  {lo * Z_ADOPT:.6f}  {hi * Z_ADOPT:.6f}')
    path.write_text('\n'.join(lines) + '\n')
    print(f'Saved {path}')


DATASETS = [
    (r'$\rm VdB01\,/\,C\,IV$',    'VDB', 'C4', 'o', -0.045),
    (r'$\rm C\&S03\,/\,C\,IV$',   'CSH', 'C4', 's',  0.000),
    (r'$\rm C\&S03\,/\,O\,III]$', 'CSH', 'O3', '^',  0.045),
]


def figure3(all_res, radii, out='plots/nitrogen/Nitrogen_abundance_vs_rBLR_loc.png',
            datasets=DATASETS, primary='VDB/C4', fit_label=r'VdB01/C\,IV'):
    """Two-axis gradient figure: N/O (left), N/H at Z_alpha = 3 (right), with the
    given (composite, denominator) datasets and the MCMC power-law fit to the
    `primary` set (lower limits never enter the fit)."""
    res = {i: v for i, v in all_res[primary].items() if not v[3]}
    ions = [i for i in IONS if i in res]
    log_r = [np.log10(radii[i][0]) for i in ions]
    log_z = [np.log10(res[i][0] * Z_ADOPT) for i in ions]
    # Centroid systematics (dex) from varying the LOC weights (Gamma, beta), NOT
    # the 25-75% emission widths drawn as bars: N V is anchored to the NGC 5548
    # lag; the outer-ion centroids move ~+-0.3 dex.  See joint_gradient_fit.SIG_R.
    SIG_R_CENTROID = {'N5': 0.10, 'N4': 0.30, 'N3': 0.30}
    sig_r = [SIG_R_CENTROID[i] for i in ions]
    sig_z = [max(0.5 * (np.log10(res[i][2]) - np.log10(res[i][1])), 0.02)
             for i in ions]

    samples = fit_powerlaw_mcmc(log_r, log_z, sig_r, sig_z)
    m_s, b_s = samples[:, 0], samples[:, 1]
    LN10 = np.log(10.0)
    g_lo, g_c, g_hi = np.percentile(m_s / LN10, [16, 50, 84])
    m_lo, m_med, m_hi = np.percentile(m_s, [16, 50, 84])
    print(f'  LOC power-law slope ({primary}) m = {m_med:.3f} '
          f'(+{m_hi - m_med:.3f}, -{m_med - m_lo:.3f})')

    fig, ax = plt.subplots(figsize=(9.5, 7))
    r_min = min(radii[i][1] for i in ions)
    r_max = max(radii[i][2] for i in ions)
    r_fit = np.logspace(np.log10(r_min) - 0.1, np.log10(r_max) + 0.1, 300)
    # fit band drawn on the N/O axis: X = (N/H)/Z_ADOPT (constant shift in log)
    pred = b_s[:, None] + m_s[:, None] * np.log10(r_fit)[None, :] - np.log10(Z_ADOPT)
    z_lo, z_mid, z_hi = (10 ** np.percentile(pred, q, axis=0) for q in (16, 50, 84))
    ax.fill_between(r_fit, z_lo, z_hi, color='dodgerblue', alpha=0.20, zorder=1)
    ax.plot(r_fit, z_mid, color='dodgerblue', lw=2.5, zorder=2)

    for label, skey, den, marker, dx in datasets:
        key = f'{skey}/{den}'
        dres = all_res[key]
        for ion in IONS:
            if ion not in dres:
                continue
            x, lo, hi, off = dres[ion]
            r, r_lo, r_hi = radii[ion]
            rp = r * 10 ** dx
            ax.errorbar(rp, x,
                        xerr=[[rp - r_lo], [r_hi - rp]] if key == primary else None,
                        yerr=[[max(x - lo, 0)], [max(hi - x, 0)]],
                        fmt=marker, ms=11, mfc=ION_COLOR[ion], mec='none',
                        ecolor=ION_COLOR[ion],
                        elinewidth=1.8, capsize=4, zorder=4)

    ann = (r'$k_{\rm N} \equiv \dfrac{d\log\,({\rm N/H})}{d\log r} = '
           r'%.2f^{+%.2f}_{-%.2f}$' % (m_med, m_hi - m_med, m_med - m_lo))
    ax.text(0.03, 0.05, ann, transform=ax.transAxes, fontsize=13, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.4', lw=1.2))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.05, 30)
    ax.set_xlabel(r'$r_{\rm BLR}~[\rm ld]$')
    ax.set_ylabel(r'$\rm (N/\alpha)/(N/\alpha)_{\odot}$')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=False, right=False,
                   length=9, width=1.5, direction='in')
    ax.tick_params(axis='both', which='minor', top=False, right=False,
                   length=4, width=1, direction='in')

    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(ax.get_ylim()[0] * Z_ADOPT, ax.get_ylim()[1] * Z_ADOPT)
    ax2.set_ylabel(r'$\rm (N/H)/(N/H)_{\odot}~at~\it{Z_\alpha}\rm =3\,\it{Z_\odot}$',
                   rotation=270, labelpad=30)
    ax2.minorticks_on()
    ax2.tick_params(axis='y', which='major', length=9, width=1.5, direction='in')
    ax2.tick_params(axis='y', which='minor', length=4, width=1, direction='in')

    ax_top = ax.twiny()
    ax_top.set_xlim(min(ION_POT.values()), max(ION_POT.values()))
    ax_top.invert_xaxis()
    ax_top.set_xlabel(r'$\rm Ionization~Potential~[eV]$', labelpad=10)
    ax_top.minorticks_on()
    ax_top.tick_params(top=True, which='major', length=9, width=1.5, direction='in')
    ax_top.tick_params(top=True, which='minor', length=4, width=1, direction='in')

    ion_handles = [Line2D([], [], marker='s', ls='', ms=11, mfc=ION_COLOR[i],
                          mec='none', label=ION_LABEL[i]) for i in IONS]
    set_handles = [Line2D([], [], marker=m, ls='', ms=11, mfc='0.6', mec='none',
                          label=lab) for lab, _, _, m, _ in datasets]
    leg1 = ax.legend(handles=ion_handles, loc='upper right', fontsize=14,
                     frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=set_handles, loc='lower left', fontsize=14, frameon=False,
              bbox_to_anchor=(0.0, 0.12))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200,
                bbox_extra_artists=[ax2.yaxis.label, ax_top.xaxis.label])
    plt.close(fig)
    print(f'Saved {out}')


def figure6(out='plots/alpha/loc_gradient_noA_Qref58/Z_profiles_population_loc.png'):
    import pandas as pd
    import sys
    sys.path.insert(0, '.')
    from alpha.plot_Z_profiles import (plot_Z_profiles_population, _drop_railed,
                                       _logQ_per_object, LOG_Q_REF)
    csv = 'fits/alpha/loc_gradient_noA_Qref58/loc_fit_summary.csv'
    df = pd.read_csv(csv).dropna(
        subset=['k', 'log_r_ref', 'log10_Z_norm', 'z']).reset_index(drop=True)
    df['logQ_med'] = _logQ_per_object(df['rm_id'].tolist()).values
    df['log_r_blr'] = df['log_r_ref'] + df['p'] * (df['logQ_med'] - LOG_Q_REF)
    df = _drop_railed(df)
    print(f'  {len(df)} objects after dropping railed fits')
    plot_Z_profiles_population(df, out, nitrogen_path=str(OUT_DAT))


def main():
    radii = loc_radii()
    all_res = {'VDB/C4': infer_loc(VDB, 'C4'),
               'CSH/C4': infer_loc(CSH, 'C4'),
               'CSH/O3': infer_loc(CSH, 'O3')}
    for tag, res in all_res.items():
        print(f'  -- {tag} --')
        for ion, (x, lo, hi, off) in res.items():
            lim = ' (limit)' if off else ''
            print(f'    {ion}: X = {x:.2f} [{lo:.2f}, {hi:.2f}]'
                  f'  ->  N/H = {x * Z_ADOPT:.2f}{lim}')
    write_dat(all_res['VDB/C4'], radii)          # primary set feeds Figure 6
    figure3(all_res, radii)
    figure3(all_res, radii,
            out='plots/nitrogen/Nitrogen_abundance_vs_rBLR_loc_cs03.png',
            datasets=[d for d in DATASETS if d[1] == 'CSH'],
            primary='CSH/C4', fit_label=r'C\&S03/C\,IV')
    figure6()


if __name__ == '__main__':
    main()
