#!/usr/bin/env python3
"""Joint nitrogen-gradient fit using both C&S03 denominators (C IV and O III]).

Model:  log10 X_ij = b + m log10 r_i + delta * [j = OIII]]  (+ intrinsic scatter)

where X is the nitrogen scale factor (N relative to the alpha-elements) inferred
from the LOC model at Z_alpha = 3 (metals-3 normalisation, metals-1 shape).
delta absorbs the systematic offset between the two denominators, which is
either a real C/O deviation from solar or a model error in the O III]/C IV
emissivity ratio; fitting it explicitly keeps it out of the slope.

Error structure:
- Each observed ratio error is propagated through the local slope of the
  inversion curve, dlogX/dlogR.
- The numerator flux error is COMMON to the C IV- and O III]-based points of the
  same ion, so those pairs are correlated in the measurement covariance.
- The two points of a pair share the same radius, so the m^2 sigma_r^2 term is
  fully correlated within a pair as well.
- A free intrinsic scatter s (dex) is added to the diagonal.

The N V / O III] point is a lower limit (off the model grid) and is excluded
from the likelihood; the tension between it and the posterior prediction is
reported at the end.

Data: Constantin & Shields (2003) composite only, so numerator fluxes are shared
between the two denominators by construction.
"""
import numpy as np

from loc_nitrogen_abundance import load_grid, VDB, CSH   # noqa: F401  (VDB unused)
from plot_loc_nitrogen_composites import curve_at_z3

Z_ADOPT = 3.0

# Abundance-consistent LOC emission radii (same phi/n range and matter-bounded
# cut as the inference; N V anchored to the NGC 5548 lag).
from loc_nitrogen_abundance import consistent_loc_radii   # noqa: E402
RADII = consistent_loc_radii()

# (ion, denominator) points entering the fit; N5/O3 is a lower limit -> excluded
POINTS = [('N5', 'C4'), ('N4', 'C4'), ('N3', 'C4'), ('N4', 'O3'), ('N3', 'O3')]


def frac_err(src, key):
    v, e = src[key]
    return e / v


def build_data():
    """Return per-point (log_r, log_X, sig_num_logX, sig_den_logX, ion, den)."""
    rows = []
    for ion, den in POINTS:
        r_obs = CSH[ion][0] / CSH[den][0]
        xs, ys = curve_at_z3(ion, den)
        # local inversion slope dlogX/dlogR at the solution
        x0 = float(np.interp(r_obs, ys[np.argsort(ys)], xs[np.argsort(ys)]))
        eps = 0.02
        r_hi, r_lo_ = r_obs * 10 ** eps, r_obs * 10 ** -eps
        x_hi = float(np.interp(r_hi, ys[np.argsort(ys)], xs[np.argsort(ys)]))
        x_lo = float(np.interp(r_lo_, ys[np.argsort(ys)], xs[np.argsort(ys)]))
        slope = (np.log10(x_hi) - np.log10(x_lo)) / (2 * eps)
        s_num = slope * frac_err(CSH, ion) / np.log(10)
        s_den = slope * frac_err(CSH, den) / np.log(10)
        rows.append(dict(ion=ion, den=den, log_r=np.log10(RADII[ion][0]),
                         log_x=np.log10(x0), s_num=s_num, s_den=s_den))
        print(f'  {ion}/{den}: X = {x0:.2f}  dlogX/dlogR = {slope:.2f}  '
              f'sig_num = {s_num:.3f}  sig_den = {s_den:.3f}')
    return rows


# Radius (centroid) systematics in dex.  These are NOT the 25-75% widths of the
# LOC emission profiles (which describe the radial extent of the emission, not
# the uncertainty of its centroid); they come from varying the LOC weights over
# Gamma, beta in [-1.5, -0.5]: the N V centroid is anchored to the NGC 5548 lag,
# while the N IV] and N III] centroids move by ~+-0.3 dex, nearly in lockstep
# (their ratio is stable), so the outer-ion systematic is strongly correlated.
SIG_R = {'N5': 0.10, 'N4': 0.30, 'N3': 0.30}
RHO_R_OUTER = 0.9        # N IV] - N III] centroid correlation


def covariance(rows, m, s_int):
    n = len(rows)
    C = np.zeros((n, n))
    for i, ri in enumerate(rows):
        sr_i = SIG_R[ri['ion']]
        C[i, i] = ri['s_num'] ** 2 + ri['s_den'] ** 2 + (m * sr_i) ** 2 + s_int ** 2
        for j in range(i + 1, n):
            rj = rows[j]
            sr_j = SIG_R[rj['ion']]
            cov = 0.0
            if ri['ion'] == rj['ion']:
                # shared numerator flux and shared (identical) radius
                cov = ri['s_num'] * rj['s_num'] + (m * sr_i) * (m * sr_j)
            elif ri['ion'] in ('N3', 'N4') and rj['ion'] in ('N3', 'N4'):
                # lever-arm systematic: the two outer centroids move together
                cov = RHO_R_OUTER * (m * sr_i) * (m * sr_j)
            C[i, j] = C[j, i] = cov
    return C


def log_post(p, rows):
    m, b, delta, lns = p
    if not (-2.0 < m < 0.5 and -3.0 < b < 3.0 and -1.0 < delta < 1.0
            and -12.0 < lns < 1.5):
        return -np.inf
    s_int = np.exp(lns)
    C = covariance(rows, m, s_int)
    resid = np.array([r['log_x'] - (b + m * r['log_r']
                                    + (delta if r['den'] == 'O3' else 0.0))
                      for r in rows])
    try:
        Cinv = np.linalg.inv(C)
        _, logdet = np.linalg.slogdet(C)
    except np.linalg.LinAlgError:
        return -np.inf
    return -0.5 * (resid @ Cinv @ resid + logdet)


def run_fit(n_steps=200000, burn=0.3, thin=10, seed=4):
    print('Building data points (C&S03, LOC at Z_alpha = 3):')
    rows = build_data()

    rng = np.random.default_rng(seed)
    p = np.array([-0.8, 0.5, 0.15, np.log(0.15)])
    lp = log_post(p, rows)
    step = np.array([0.08, 0.10, 0.06, 0.25])
    chain = np.empty((n_steps, 4))
    for i in range(n_steps):
        q = p + step * rng.standard_normal(4)
        lq = log_post(q, rows)
        if lq - lp > np.log(rng.random()):
            p, lp = q, lq
        chain[i] = p
    return chain[int(burn * n_steps)::thin], rows


def figure(samp, out='plots/nitrogen/Nitrogen_abundance_vs_rBLR_loc_cs03_joint.png'):
    """C&S03-only gradient figure with the JOINT fit drawn on both denominator
    scales: grey band + solid line = C IV scale, dashed line = O III] scale
    (shifted up by delta)."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from plot_loc_nitrogen_composites import infer

    plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                         'font.family': 'serif', 'font.weight': 'heavy',
                         'font.size': 20})
    plt.rcParams['text.latex.preamble'] = (r'\usepackage{amsmath} '
                                           r'\usepackage{bm} \boldmath')

    ION_COLOR = {'N3': '#5B2B7B', 'N4': '#C2185B', 'N5': '#E67E22'}
    ION_LABEL = {'N3': r'$\rm N\,III]$', 'N4': r'$\rm N\,IV]$',
                 'N5': r'$\rm N\,V$'}
    ION_POT = {'N3': 47.4, 'N4': 77.5, 'N5': 97.9}
    DSETS = [(r'$\rm CS03\,/\,C\,IV$', 'C4', 's', 0.000),
             (r'$\rm CS03\,/\,O\,III]$', 'O3', '^', 0.045)]

    m_s, b_s, d_s = samp[:, 0], samp[:, 1], samp[:, 2]
    m_lo, m_c, m_hi = np.percentile(m_s, [16, 50, 84])
    d_lo, d_c, d_hi = np.percentile(d_s, [16, 50, 84])

    fig, ax = plt.subplots(figsize=(9.5, 7))
    r_fit = np.logspace(np.log10(RADII['N5'][1]) - 0.1,
                        np.log10(RADII['N3'][2]) + 0.1, 300)
    pred_c4 = b_s[:, None] + m_s[:, None] * np.log10(r_fit)[None, :]
    z_lo, z_mid, z_hi = (10 ** np.percentile(pred_c4, q, axis=0)
                         for q in (16, 50, 84))
    ax.fill_between(r_fit, z_lo, z_hi, color='orangered', alpha=0.18, zorder=1)
    ax.plot(r_fit, z_mid, color='orangered', lw=2.5, zorder=2,
            solid_capstyle='round',
            label=r'$\rm joint~fit~(C\,IV~points)$')
    pred_o3_mid = 10 ** np.percentile(pred_c4 + d_s[:, None], 50, axis=0)
    ax.plot(r_fit, pred_o3_mid, color='orangered', lw=2.5, ls='--', zorder=2,
            dash_capstyle='round',
            label=r'$\rm joint~fit~(O\,III]~points)$')

    # Z_alpha systematic: factor by which the inferred X moves if the assumed
    # background metallicity is 1 or 10 Zsun instead of the adopted 3 Zsun.
    # Computed from the separable LOC model (solve_N_and_Z), applied as a
    # multiplicative range to each plotted point.
    import numpy as _np
    import solve_N_and_Z as S
    from loc_nitrogen_abundance import load_grid as _load_grid, invert as _invert
    S.GRID_X = _load_grid('N25_m1')
    S.GRID_Z = S.load_varym()
    _gx = _np.logspace(_np.log10(0.05), _np.log10(9.0), 400)

    from loc_nitrogen_abundance import obs_ratio as _obs_ratio

    def z_factors(ion, den):
        r_obs, _ = _obs_ratio(CSH, ion, den)
        vals = {}
        for z in (1.0, 3.0, 10.0):
            model, _ = S.build_model(ion, den, _gx, _np.full_like(_gx, z))
            vals[z] = _invert(_gx, model, r_obs)[0]
        return vals[1.0] / vals[3.0], vals[10.0] / vals[3.0]

    for label, den, marker, dx in DSETS:
        for ion in ('N5', 'N4', 'N3'):
            x, lo, hi, off = infer(ion, CSH, den)
            r, r_lo, r_hi = RADII[ion]
            rp = r * 10 ** dx
            f1, f10 = z_factors(ion, den)
            z_lo_pt, z_hi_pt = sorted((x * f1, x * f10))
            eb = ax.errorbar(rp, x,
                             yerr=[[max(x - z_lo_pt, 0)],
                                   [max(z_hi_pt - x, 0)]],
                             fmt='none', ecolor=ION_COLOR[ion], elinewidth=6,
                             alpha=0.25, capsize=0, zorder=3)
            for lc in eb[2]:
                lc.set_capstyle('round')
            ax.errorbar(rp, x,
                        xerr=[[rp - r_lo], [r_hi - rp]] if den == 'C4' else None,
                        yerr=[[max(x - lo, 0)], [max(hi - x, 0)]],
                        fmt=marker, ms=11, mfc=ION_COLOR[ion], mec='none',
                        ecolor=ION_COLOR[ion],
                        elinewidth=1.8, capsize=4, zorder=4)

    ann = (r'$k_{\rm N} \equiv \dfrac{d\log\,({\rm N/H})}{d\log r} = '
           r'%.2f^{+%.2f}_{-%.2f}$' % (m_c, m_hi - m_c, m_c - m_lo)
           + '\n'
           + r'$\delta = %.2f^{+%.2f}_{-%.2f}\,{\rm dex}$'
           % (d_c, d_hi - d_c, d_c - d_lo))
    ax.text(0.03, 0.05, ann, transform=ax.transAxes, fontsize=13, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.4', lw=1.2))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(0.05, 30)
    ax.set_xlabel(r'$r~[\rm ld]$')
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
    ax_top.tick_params(top=True, which='major', length=9, width=1.5,
                       direction='in')
    ax_top.tick_params(top=True, which='minor', length=4, width=1,
                       direction='in')

    ion_handles = [Line2D([], [], marker='s', ls='', ms=11, mfc=ION_COLOR[i],
                          mec='none', label=ION_LABEL[i])
                   for i in ('N5', 'N4', 'N3')]
    set_handles = ([Line2D([], [], marker=m, ls='', ms=11, mfc='0.6', mec='none',
                           label=lab) for lab, _, m, _ in DSETS]
                   + [Line2D([], [], marker='|', ls='', color='0.5',
                             ms=18, mew=6, alpha=0.25,
                             label=r'$Z_{\alpha} = 1{-}10\,Z_{\odot}$'),
                      Line2D([], [], color='orangered', lw=2.5,
                             label=r'$\rm joint~fit~(C\,IV~points)$'),
                      Line2D([], [], color='orangered', lw=2.5, ls='--',
                             label=r'$\rm joint~fit~(O\,III]~points)$')])
    leg1 = ax.legend(handles=ion_handles, loc='upper right', fontsize=14,
                     frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=set_handles, loc='lower left', fontsize=13,
              frameon=False, bbox_to_anchor=(0.0, 0.28))

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200,
                bbox_extra_artists=[ax2.yaxis.label, ax_top.xaxis.label])
    plt.close(fig)
    print(f'Saved {out}')


def main():
    import argparse
    import plot_loc_nitrogen_composites as P
    from loc_nitrogen_abundance import radii_chunks
    global RADII
    ap = argparse.ArgumentParser(description='Joint nitrogen-gradient fit')
    ap.add_argument('--norm-grid', default=None,
                    help='metals-3 normalisation grid (a key of GRIDS, e.g. N23_m3_turb0); '
                         'default N23_m3')
    ap.add_argument('--radii-from-norm', action='store_true',
                    help='also take the emission radii from the normalisation grid '
                         '(default: metals-3 N23 grid, N23_m3)')
    ap.add_argument('--no-figure', action='store_true', help='skip the gradient figure')
    args = ap.parse_args()
    if args.norm_grid:
        P.use_norm_grid(args.norm_grid)
    if args.radii_from_norm:
        RADII = consistent_loc_radii(chunks=radii_chunks(args.norm_grid or 'N23_m3'))
    print(f'Normalisation grid: {args.norm_grid or "N23_m3"}; radii from '
          f'{args.norm_grid or "N23_m3" if args.radii_from_norm else "N23_m3"}: '
          + ', '.join(f'{i} {RADII[i][0]:.2f} ld' for i in ('N5', 'N4', 'N3')))

    samp, rows = run_fit()

    labels = ['slope m', 'b (logX at 1 ld, CIV scale)', 'delta (OIII]-CIV)',
              'ln s']
    print('\nJoint-fit posterior (median, 16-84%):')
    for k, lab in enumerate(labels):
        lo, med, hi = np.percentile(samp[:, k], [16, 50, 84])
        print(f'  {lab:>30}: {med:+.3f}  (+{hi - med:.3f}, -{med - lo:.3f})')

    m_s, b_s, d_s = samp[:, 0], samp[:, 1], samp[:, 2]
    g = np.percentile(m_s / np.log(10), [16, 50, 84])
    print(f'\n  dlog(N/H)/dr at 1 ld: {g[1]:.3f} (+{g[2] - g[1]:.3f}, '
          f'-{g[1] - g[0]:.3f}) dex/ld')

    # absolute values at the N V radius (2 ld), on both denominator scales
    for tag, off in (('C IV scale (N/C)', 0.0), ('O III] scale (N/O)', 1.0)):
        pred = b_s + m_s * np.log10(2.0) + off * d_s
        lo, med, hi = 10 ** np.percentile(pred, [16, 50, 84])
        print(f'  X at 2 ld, {tag}: {med:.2f} [{lo:.2f}, {hi:.2f}]'
              f'  ->  N/H = {med * Z_ADOPT:.2f}')

    # the excluded N V / O III] point (extrapolated beyond the model grid)
    from plot_loc_nitrogen_composites import infer
    x_ext, lo_ext, hi_ext, _ = infer('N5', CSH, 'O3')
    pred_nv_o3 = b_s + m_s * np.log10(2.0) + d_s
    frac = float(np.mean(10 ** pred_nv_o3 > x_ext))
    print(f'\n  N V/O III] (extrapolated, excluded from fit): '
          f'X = {x_ext:.1f} [{lo_ext:.1f}, {hi_ext:.1f}]; '
          f'posterior P(X > {x_ext:.1f}) = {frac:.2f}')

    if not args.no_figure:
        figure(samp)


if __name__ == '__main__':
    main()
