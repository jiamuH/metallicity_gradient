#!/usr/bin/env python3
"""Nitrogen and alpha-element radial profiles on one panel (new Fig. 6).

Shows the SAME points and the SAME fit as the reference gradient figure
(plots/nitrogen/Nitrogen_abundance_vs_rBLR_loc_cs03_joint.png), both taken from
nitrogen_result.py, with the population alpha-element profile overlaid:

  - six Constantin & Shields points (squares = ratios to C IV, triangles = to
    O III] 1663), pale bars for the Z_alpha = 1-10 Zsun systematic;
  - the joint fit, drawn on both reference-line scales (solid = C IV points,
    dashed = O III] points, offset by the fitted delta);
  - the alpha-element median and 16-84% object-to-object spread from the
    SDSS-RM LOC + breathing fits (Sec. 3).

Both abundances are relative to solar, with N/H = X * Z_alpha at the adopted
Z_alpha = 3 from the alpha-element analysis.

Output: plots/nitrogen/Z_profiles_N_alpha_loc.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, '.')
from alpha.plot_Z_profiles import (_drop_railed, _logQ_per_object, LOG_Q_REF,
                                   LOGX_LO, LOGX_HI, LD_PER_CM)      # noqa: E402
import nitrogen_result as NR                                          # noqa: E402

plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2.8,
                     'font.family': 'serif', 'font.weight': 'heavy',
                     'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

N_COLOR, CNO_COLOR = 'orangered', 'dodgerblue'
CSV = 'fits/alpha/loc_gradient_noA_Qref58/loc_fit_summary.csv'
R_MAX = 100.0
OUT = 'plots/nitrogen/Z_profiles_N_alpha_loc.png'


def alpha_profile():
    """(r_ld, median, lo16, hi84) of Z_alpha(r) over the population."""
    df = pd.read_csv(CSV).dropna(
        subset=['k', 'log_r_ref', 'log10_Z_norm', 'z']).reset_index(drop=True)
    df['logQ_med'] = _logQ_per_object(df['rm_id'].tolist()).values
    df['log_r_blr'] = df['log_r_ref'] + df['p'] * (df['logQ_med'] - LOG_Q_REF)
    df = _drop_railed(df)
    print(f'  {len(df)} objects in the alpha-element stack')
    global K_CNO_FILTERED
    K_CNO_FILTERED = float(np.median(df['k'].values))

    lrb = df['log_r_blr'].values
    lo = np.nanmin(lrb) + LOGX_LO - 0.05
    hi = np.nanmax(lrb) + LOGX_HI + 0.05
    logr_cm = np.linspace(lo, hi, int(round((hi - lo) / 0.1)) + 1)
    r_ld = 10 ** logr_cm * LD_PER_CM

    logZ = (df['log10_Z_norm'].values[:, None]
            + df['k'].values[:, None] * (logr_cm[None, :]
                                         - df['log_r_ref'].values[:, None]))
    mask = ((logr_cm[None, :] >= lrb[:, None] + LOGX_LO)
            & (logr_cm[None, :] <= lrb[:, None] + LOGX_HI))
    med = np.full(logr_cm.size, np.nan)
    lo16 = np.full(logr_cm.size, np.nan)
    hi84 = np.full(logr_cm.size, np.nan)
    for i in range(logr_cm.size):
        v = logZ[mask[:, i], i]
        if v.size >= 10:
            med[i], lo16[i], hi84[i] = np.percentile(v, [50, 16, 84])
    from scipy.ndimage import gaussian_filter1d
    ok = np.isfinite(med)
    if ok.any():
        w = np.maximum(gaussian_filter1d(ok.astype(float), 1.0, mode='nearest'),
                       1e-9)
        for arr in (med, lo16, hi84):
            tmp = np.where(ok, arr, 0.0)
            arr[ok] = (gaussian_filter1d(tmp, 1.0, mode='nearest') / w)[ok]
    return r_ld, 10 ** med, 10 ** lo16, 10 ** hi84


SC_COLOR = '#7A0019'     # self-consistent overlay (radii from the metals-3 grid)


def main(out=OUT, style='two-scale', overlay_selfconsistent=False):
    """style selects the fit presentation:
    'two-scale' : solid C IV line, dashed O III] line, band on the C IV scale
                  (as in the reference gradient figure);
    'unified'   : everything on the N/O scale -- the C IV points are converted
                  with the fitted delta (N/C * 10^delta = N/O), ONE line + band
                  for the posterior of (b + delta) + k_N log r, delta's
                  uncertainty folded into the band;
    'envelope'  : both lines kept, ONE shade spanning from the C IV band's
                  lower edge to the O III] band's upper edge (fit uncertainty
                  and scale offset combined in a single region)."""
    unified = style == 'unified'
    print('Alpha-element population profile:')
    r_a, z_med, z_lo, z_hi = alpha_profile()
    print(f'Nitrogen points and joint fit (canonical, Z_alpha = {NR.Z_ADOPT:.0f}):')
    pts = NR.points()
    samp, summ = NR.fit()
    m_s, b_s, d_s = samp[:, 0], samp[:, 1], samp[:, 2]
    d_med = summ['delta']
    k_cno = K_CNO_FILTERED
    print(f'  k_N = {summ["k"]:.2f} (+{summ["k_hi"] - summ["k"]:.2f}, '
          f'-{summ["k"] - summ["k_lo"]:.2f})   delta = {d_med:.2f}   '
          f'<k>_CNO = {k_cno:.2f}')

    fig, ax = plt.subplots(figsize=(11, 6.2))

    # alpha-element population profile
    m = (r_a <= R_MAX) & np.isfinite(z_med)
    ax.fill_between(r_a[m], z_lo[m], z_hi[m], color=CNO_COLOR, alpha=0.18,
                    zorder=1)
    ax.plot(r_a[m], z_med[m], '-', color=CNO_COLOR, lw=3, zorder=3,
            solid_capstyle='round')

    # joint fit, converted to N/H (X * Z_alpha)
    r_fit = NR.fit_range(pts)
    base = b_s[:, None] + m_s[:, None] * np.log10(r_fit)[None, :] \
        + np.log10(NR.Z_ADOPT)
    if style == 'unified':
        # single band on the N/O scale; delta's posterior widens the band
        pred = base + d_s[:, None]
        n_lo, n_mid, n_hi = (10 ** np.percentile(pred, q, axis=0)
                             for q in (16, 50, 84))
        ax.fill_between(r_fit, n_lo, n_hi, color=N_COLOR, alpha=0.18, zorder=1)
        ax.plot(r_fit, n_mid, '-', color=N_COLOR, lw=3, zorder=3,
                solid_capstyle='round')
    elif style == 'envelope':
        # one shade covering both scales; both central lines kept
        n_lo = 10 ** np.percentile(base, 16, axis=0)
        n_hi = 10 ** np.percentile(base + d_s[:, None], 84, axis=0)
        n_mid = 10 ** np.percentile(base, 50, axis=0)
        ax.fill_between(r_fit, n_lo, n_hi, color=N_COLOR, alpha=0.18, zorder=1)
        ax.plot(r_fit, n_mid, '-', color=N_COLOR, lw=3, zorder=3,
                solid_capstyle='round')
        n_o3 = 10 ** np.percentile(base + d_s[:, None], 50, axis=0)
        ax.plot(r_fit, n_o3, '--', color=N_COLOR, lw=3, zorder=3,
                dash_capstyle='round')
    else:
        n_lo, n_mid, n_hi = (10 ** np.percentile(base, q, axis=0)
                             for q in (16, 50, 84))
        ax.fill_between(r_fit, n_lo, n_hi, color=N_COLOR, alpha=0.18, zorder=1)
        ax.plot(r_fit, n_mid, '-', color=N_COLOR, lw=3, zorder=3,
                solid_capstyle='round')
        n_o3 = 10 ** np.percentile(base + d_s[:, None], 50, axis=0)
        ax.plot(r_fit, n_o3, '--', color=N_COLOR, lw=3, zorder=3,
                dash_capstyle='round')

    # the six points; in unified mode the C IV points move to the N/O scale
    for _, src, den, marker, dx in NR.DATASETS:
        shift = 10 ** d_med if (unified and den == 'C4') else 1.0
        for ion in NR.IONS:
            d = pts[(ion, den)]
            rp = d['r'] * 10 ** dx
            eb = ax.errorbar(rp, d['nh'] * shift,
                             yerr=[[max((d['nh'] - d['sys_lo'] * NR.Z_ADOPT)
                                        * shift, 0)],
                                   [max((d['sys_hi'] * NR.Z_ADOPT - d['nh'])
                                        * shift, 0)]],
                             fmt='none', ecolor=NR.ION_COLOR[ion],
                             elinewidth=6, alpha=0.25, capsize=0, zorder=4)
            for lc in eb[2]:
                lc.set_capstyle('round')
            ax.errorbar(rp, d['nh'] * shift,
                        xerr=([[rp - d['r_lo']], [d['r_hi'] - rp]]
                              if den == NR.FIT_DEN else None),
                        yerr=[[max(d['nh'] - d['nh_lo'], 0) * shift],
                              [max(d['nh_hi'] - d['nh'], 0) * shift]],
                        fmt=marker, ms=11, mfc=NR.ION_COLOR[ion], mec='k',
                        mew=0.7, ecolor=NR.ION_COLOR[ion], elinewidth=1.8,
                        capsize=4, zorder=5)

    # optional overlay: the same abundances placed at radii from the metals-3
    # grid (the one the abundances come from), with the joint fit redone there
    if overlay_selfconsistent:
        from loc_nitrogen_abundance import consistent_loc_radii, radii_chunks
        radii_sc = consistent_loc_radii(chunks=radii_chunks('N23_m3'))
        pts_sc = NR.points(radii=radii_sc)
        samp_sc, summ_sc = NR.fit(radii=radii_sc)
        print(f'  self-consistent radii: k_N = {summ_sc["k"]:.2f} '
              f'(+{summ_sc["k_hi"] - summ_sc["k"]:.2f}, -{summ_sc["k"] - summ_sc["k_lo"]:.2f})')
        r_sc = NR.fit_range(pts_sc)
        base_sc = samp_sc[:, 1:2] + samp_sc[:, 0:1] * np.log10(r_sc)[None, :] \
            + np.log10(NR.Z_ADOPT)
        ax.plot(r_sc, 10 ** np.percentile(base_sc, 50, axis=0), ':', color=SC_COLOR,
                lw=3, zorder=6)
        ax.plot(r_sc, 10 ** np.percentile(base_sc + samp_sc[:, 2:3], 50, axis=0), ':',
                color=SC_COLOR, lw=3, zorder=6)
        for _, src, den, marker, dx in NR.DATASETS:
            for ion in NR.IONS:
                d = pts_sc[(ion, den)]
                ax.plot(d['r'] * 10 ** dx, d['nh'], marker, ms=11, mfc='none',
                        mec=SC_COLOR, mew=2.0, zorder=6)

    # ---- axes: dual colour-coded scale, as in the published Fig. 6 ----
    ax.set_xscale('log')
    ax.set_yscale('log')
    vals = ([pts[k]['nh_lo'] for k in pts] + [pts[k]['nh_hi'] for k in pts]
            + [z_lo[m].min(), z_hi[m].max(), n_lo.min(), n_hi.max()])
    ax.set_ylim(min(vals) / 2.5, max(vals) * 4.0)
    ax.set_xlabel(r'$r~[{\rm ld}]$', fontsize=22)
    ax.set_ylabel(r'$({\rm N/H})/({\rm N/H})_{\odot}$', color=N_COLOR,
                  fontsize=20)

    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(*ax.get_ylim())
    ax2.set_ylabel(r'$[{\rm (C+N+O)/H}]/[{\rm (C+N+O)/H}]_{\odot}$',
                   rotation=270, labelpad=26, color=CNO_COLOR, fontsize=19)
    ax2.minorticks_on()
    ax2.tick_params(axis='y', which='major', length=10, width=1.5,
                    direction='in', pad=5, labelsize=20)
    ax2.tick_params(axis='y', which='minor', length=5, width=1, direction='in')
    ax2.tick_params(axis='y', which='both', colors=CNO_COLOR)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True, length=10,
                   width=1.5, direction='in', pad=5, labelsize=20)
    ax.tick_params(axis='both', which='minor', top=True, right=True, length=5,
                   width=1, direction='in')
    ax.tick_params(axis='y', which='both', right=False, colors=N_COLOR)
    ax.spines['left'].set_color(N_COLOR)
    ax2.spines['left'].set_color(N_COLOR)
    ax.spines['right'].set_color(CNO_COLOR)
    ax2.spines['right'].set_color(CNO_COLOR)

    # ---- legends: species row, then slopes, as in the published Fig. 6 ----
    sp_handles = [Line2D([0], [0], marker='o', color='w', mfc=NR.ION_COLOR[i],
                         mec='k', ms=9, label=NR.ION_LABEL[i]) for i in NR.IONS]
    leg_sp = ax.legend(handles=sp_handles, loc='upper right',
                       bbox_to_anchor=(1.0, 1.0), ncol=3, columnspacing=1.2,
                       handletextpad=0.4, fontsize=18, frameon=False)
    ax.add_artist(leg_sp)
    k_handles = [
        Line2D([0], [0], color=CNO_COLOR, lw=2.6,
               label=rf'$\langle k\rangle_{{\rm CNO}} = {k_cno:+.2f}$'),
        Line2D([0], [0], color=N_COLOR, lw=2.4,
               label=rf'$k_{{\rm N}} = {summ["k"]:+.2f}$'),
    ]
    if overlay_selfconsistent:
        k_handles.append(Line2D([0], [0], color=SC_COLOR, lw=3, ls=':',
                                label=rf'$k_{{\rm N}} = {summ_sc["k"]:+.2f}~(Z_\alpha{{=}}3~\rm radii)$'))
    leg_k = ax.legend(handles=k_handles, loc='upper right',
                      bbox_to_anchor=(1.0, 0.90), fontsize=18, frameon=False)
    ax.add_artist(leg_k)
    set_labels = {lab: lab for lab, _, _, _, _ in NR.DATASETS}
    if unified:
        # C IV points are shown converted to the N/O scale with the fitted delta
        c4_lab = NR.DATASETS[0][0]
        set_labels[c4_lab] = c4_lab + r'$\rm ~(\times 10^{\delta})$'
    ref_handles = (
        [Line2D([0], [0], marker=mk, color='w', mfc='0.55', mec='k', ms=9,
                label=set_labels[lab]) for lab, _, _, mk, _ in NR.DATASETS]
        + [Line2D([0], [0], marker='|', ls='', color='0.5', ms=16, mew=5,
                  alpha=0.3, label=r'$Z_{\alpha} = 1{-}10\,Z_{\odot}$')])
    ax.legend(handles=ref_handles, loc='lower left', fontsize=17,
              frameon=False, handletextpad=0.4)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight', bbox_extra_artists=[ax2.yaxis.label])
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    if '--overlay-selfconsistent' in sys.argv:            # envelope figure + Z_alpha = 3 radii overlay only
        main(out='plots/nitrogen/Z_profiles_N_alpha_loc_envelope_selfconsistent.png',
             style='envelope', overlay_selfconsistent=True)
        sys.exit()
    main()                                                # two-scale version
    main(out='plots/nitrogen/Z_profiles_N_alpha_loc_unified.png',
         style='unified')                                 # A: single N/O scale
    main(out='plots/nitrogen/Z_profiles_N_alpha_loc_envelope.png',
         style='envelope')                                # B: envelope band
