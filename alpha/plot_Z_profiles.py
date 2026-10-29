"""Diagnostic plots from `loc_fit_summary.csv` (all rm objects fitted no-free-A):

  - Z_profiles_all.png    : per-object Z(r) best-fit curves, colored by k
  - k_histogram.png       : marginal distribution of best-fit k
  - k_vs_covariates.png   : k vs (Z_norm, z, median logQ)
  - Z_profiles_zbins.png  : population-median Z(r) stacked in redshift bins

Run from repo root with:
    python3 -m alpha.plot_Z_profiles
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.weight': 'heavy',
    'font.size': 16, 'axes.linewidth': 2, 'lines.linewidth': 2,
    'axes.labelweight': 'heavy', 'text.latex.preamble': r'\usepackage{bm} \usepackage{amsmath} \boldmath',
})

CSV = 'fits/alpha/loc_gradient_noA/loc_fit_summary.csv'
OUT_DIR = 'plots/alpha/loc_gradient_noA'
LOGX_LO, LOGX_HI = -1.0, 1.0
LD_PER_CM = 1.0 / 2.59020684e15
LOG_Q_REF = 58.0          # Q_gamma reference for the breathing law r_BLR = r_ref (Q/Q_REF)^p


def _logQ_per_object(rm_ids, data_dir='data/alpha/observed_line_ratio_data'):
    """Median logQ per object (from the line-ratio files)."""
    import sys
    sys.path.insert(0, 'alpha')
    from alpha.loc_gradient_fit import load_object_data
    out = {}
    for rid in rm_ids:
        try:
            d, _ = load_object_data(rid)
            allq = np.concatenate([d[t]['logQ'] for t in d])
            out[rid] = float(np.median(allq))
        except Exception:
            out[rid] = np.nan
    return pd.Series(out)


def _style(ax):
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True, length=8, width=1.5, direction='in', pad=5)
    ax.tick_params(axis='both', which='minor', top=True, right=True, length=4, width=1, direction='in')


def plot_Z_profiles_all(df, out):
    logr_cm = np.linspace(15.0, 20.0, 280)
    r_ld = 10 ** logr_cm * LD_PER_CM
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    for _, row in df.iterrows():
        lrr, lZn, k, lrb = row['log_r_ref'], row['log10_Z_norm'], row['k'], row['log_r_blr']
        logZ = lZn + k * (logr_cm - lrr)
        mask = (logr_cm >= lrb + LOGX_LO) & (logr_cm <= lrb + LOGX_HI)
        ax.plot(r_ld[mask], 10 ** logZ[mask], '-', color=cmap(norm(k)), alpha=0.25, lw=1.2, zorder=2)
    ax.axhline(1.0, color='0.4', lw=1.0, ls=':', zorder=1)
    ax.axhline(20.0, color='0.4', lw=1.0, ls=':', zorder=1)
    medZ = np.full(logr_cm.size, np.nan)
    for i, lr in enumerate(logr_cm):
        vals = [row['log10_Z_norm'] + row['k'] * (lr - row['log_r_ref'])
                for _, row in df.iterrows()
                if row['log_r_blr'] + LOGX_LO <= lr <= row['log_r_blr'] + LOGX_HI]
        if len(vals) >= 5:
            medZ[i] = np.median(vals)
    ax.plot(r_ld, 10 ** medZ, '-', color='k', lw=2.6, zorder=4, label=r'${\rm population~median}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(r_ld.min(), r_ld.max()); ax.set_ylim(0.5, 30)
    ax.set_xlabel(r'$r~[{\rm ld}]$')
    ax.set_ylabel(r'$Z/Z_{\odot}$')
    _style(ax)
    ax.legend(loc='lower left', fontsize=12, frameon=False)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label(r'best-fit $k$', rotation=270, labelpad=18)
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig); print(f'Saved {out}')


def plot_k_histogram(df, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    k = df['k'].values
    ax.hist(k, bins=np.linspace(-1.0, 1.0, 31), color='steelblue',
            edgecolor='0.15', linewidth=1.0, alpha=0.85)
    p16, p50, p84 = np.percentile(k, [16, 50, 84])
    ax.axvline(0, color='0.4', lw=1.2, ls=':')
    ax.axvline(p50, color='crimson', lw=2.0, label=rf'median $= {p50:+.2f}$')
    ax.axvspan(p16, p84, color='crimson', alpha=0.12, label=rf'$16$--$84\%$: $[{p16:+.2f},\,{p84:+.2f}]$')
    ax.set_xlabel(r'${\rm best~fit}~k$')
    ax.set_ylabel(r'${\rm N~objects}$')
    ax.set_xlim(-1.0, 1.0)
    _style(ax)
    ax.legend(loc='upper left', fontsize=12, frameon=False)
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig); print(f'Saved {out}')


def plot_k_vs_covariates(df, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plots = [
        ('log10_Z_norm', r'$\log_{10}\,(Z_{\rm norm}/Z_{\odot})$', None),
        ('z',            r'redshift $z$',                          None),
        ('logQ_med',     r'median $\log Q~[\rm photons~s^{-1}]$',  None),
    ]
    for ax, (xkey, xlab, _) in zip(axes, plots):
        x = df[xkey].values
        y = df['k'].values
        yerr = np.vstack([df['k_lo'].values, df['k_hi'].values])
        ax.errorbar(x, y, yerr=yerr, fmt='o', ms=4, color='steelblue',
                    alpha=0.55, ecolor='0.55', elinewidth=0.7, capsize=0, zorder=2)
        ax.axhline(0, color='0.4', lw=1.0, ls=':')
        ax.set_xlabel(xlab); ax.set_ylabel(r'${\rm best~fit}~k$')
        _style(ax)
        # report Spearman rho
        from scipy.stats import spearmanr
        m = np.isfinite(x) & np.isfinite(y)
        rho, pval = spearmanr(x[m], y[m])
        ax.text(0.04, 0.96, rf'$\rho_{{\rm s}}={rho:+.2f}$, $p={pval:.2g}$',
                transform=ax.transAxes, va='top', ha='left', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.7', alpha=0.9))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig); print(f'Saved {out}')


def _load_nitrogen_points(path='nitrogen/nitrogen_abundance_vs_r.dat'):
    """Read the 6 nitrogen (N/H)/(N/H)_sun points (3 species x 2 alpha-element models)."""
    cols = ['sp', 'r', 'r_lo', 'r_hi', 'model', 'Z', 'Z_lo', 'Z_hi']
    rows = []
    with open(path) as fh:
        for ln in fh:
            if ln.startswith('#') or not ln.strip(): continue
            p = ln.split()
            rows.append(dict(zip(cols, [p[0], float(p[1]), float(p[2]), float(p[3]),
                                        p[4], float(p[5]), float(p[6]), float(p[7])])))
    return pd.DataFrame(rows)


def _combine_models(N):
    """Collapse the (species x alpha-model) rows of `N` to one point per species:
    geometric-mean abundance, y-error spanning both models' Z_low/Z_high bounds."""
    rows = []
    for sp, g in N.groupby('sp', sort=False):
        zb = 10 ** np.log10(g['Z'].values).mean()
        bounds = np.concatenate([g['Z_lo'].values, g['Z_hi'].values])
        rows.append(dict(sp=sp, r=g['r'].iloc[0], r_lo=g['r_lo'].iloc[0],
                          r_hi=g['r_hi'].iloc[0], Z=zb,
                          Z_lo=bounds.min(), Z_hi=bounds.max()))
    return pd.DataFrame(rows)


def _powerlaw_fit_N(N, n_steps=120000, burn_frac=0.3, thin=8, seed=12345):
    """Metropolis-Hastings power-law fit  log10 z = b + m log10 r  with errors in
    both axes plus an intrinsic-scatter term (Hogg, Bovy & Lang 2010 likelihood).
    Returns (n_samples, 3) array of posterior samples [m, b, ln s]."""
    log_r  = np.log10(N['r'].values)
    log_z  = np.log10(N['Z'].values)
    s_r    = 0.5 * (np.log10(N['r_hi'].values) - np.log10(N['r_lo'].values))
    s_z    = 0.5 * (np.log10(N['Z_hi'].values) - np.log10(N['Z_lo'].values))
    s2_r, s2_z = s_r ** 2, s_z ** 2
    rng = np.random.default_rng(seed)
    def log_post(p):
        m, b, lns = p
        if not (-2.0 < m < 0.5 and -3.0 < b < 3.0 and -12.0 < lns < 1.5):
            return -np.inf
        var = s2_z + (m * m) * s2_r + np.exp(2.0 * lns)
        resid = log_z - (b + m * log_r)
        return -0.5 * np.sum(resid * resid / var + np.log(var))
    m0, b0 = np.polyfit(log_r, log_z, 1)
    p = np.array([m0, b0, np.log(0.2)])
    lp = log_post(p)
    step = np.array([0.06, 0.06, 0.20])
    chain = np.empty((n_steps, 3))
    for i in range(n_steps):
        q = p + step * rng.standard_normal(3)
        lq = log_post(q)
        if np.log(rng.uniform()) < lq - lp:
            p, lp = q, lq
        chain[i] = p
    return chain[int(burn_frac * n_steps)::thin]


def plot_Z_profiles_population(df, out,
                               nitrogen_path='nitrogen/nitrogen_abundance_vs_r.dat'):
    """Population-median Z(r) with nitrogen points + power-law fit overlaid.  The radial
    grid spans both the nitrogen error-bar extent and the alpha breathing-radius windows.

    `nitrogen_path` selects the nitrogen abundance table; the default is the
    single-zone inference, and nitrogen_abundance_vs_r_loc.dat is the LOC-averaged one."""
    N = _load_nitrogen_points(nitrogen_path)
    r_lo_min, r_hi_max = N['r_lo'].min(), N['r_hi'].max()
    # radial grid spans both the nitrogen error-bar extent and the alpha breathing windows
    lrb = df['log_r_blr'].values
    lo = min(np.log10(r_lo_min / LD_PER_CM), np.nanmin(lrb) + LOGX_LO) - 0.05
    hi = max(np.log10(r_hi_max / LD_PER_CM), np.nanmax(lrb) + LOGX_HI) + 0.05
    logr_cm = np.linspace(lo, hi, int(round((hi - lo) / 0.1)) + 1)   # ~0.1-dex spacing
    r_ld = 10 ** logr_cm * LD_PER_CM
    # No extrapolation: at each r only objects whose own LOC window covers r contribute.
    # Use a higher object-count threshold so the percentiles are stable, then smooth in log r
    # to suppress bin-to-bin jitter from changing sample composition.
    MIN_OBJ, SMOOTH_BINS = 10, 1.0           # draw the population median only where >=MIN_OBJ objects contribute; smoothing sigma 1 bin
    lr_grid = logr_cm[None, :]
    lzn = df['log10_Z_norm'].values[:, None]
    k_v = df['k'].values[:, None]
    lrr = df['log_r_ref'].values[:, None]
    lrb_c = df['log_r_blr'].values[:, None]
    logZ_all = lzn + k_v * (lr_grid - lrr)
    loc_mask = (logr_cm[None, :] >= lrb_c + LOGX_LO) & (logr_cm[None, :] <= lrb_c + LOGX_HI)
    med  = np.full(logr_cm.size, np.nan)
    lo16 = np.full(logr_cm.size, np.nan)
    hi84 = np.full(logr_cm.size, np.nan)
    for i in range(logr_cm.size):
        vals = logZ_all[loc_mask[:, i], i]
        if vals.size >= MIN_OBJ:
            med[i], lo16[i], hi84[i] = np.percentile(vals, [50, 16, 84])
    # Gaussian smoothing in log r (NaN-aware: only smooth across the valid window)
    from scipy.ndimage import gaussian_filter1d
    valid = np.isfinite(med)
    if valid.any():
        wt_s = np.maximum(gaussian_filter1d(valid.astype(float), SMOOTH_BINS, mode='nearest'), 1e-9)
        for arr in (med, lo16, hi84):
            tmp = np.where(valid, arr, 0.0)
            arr[valid] = (gaussian_filter1d(tmp, SMOOTH_BINS, mode='nearest') / wt_s)[valid]

    N_COLOR, CNO_COLOR = 'orangered', 'royalblue'
    R_FIT_MAX = 100.0          # crop both fits at 100 ld
    fig, ax = plt.subplots(figsize=(8, 6))
    # alpha-element population median + 16-84 percentile band, drawn only where >=MIN_OBJ objects contribute
    amask = r_ld <= R_FIT_MAX
    ax.fill_between(r_ld[amask], (10 ** lo16)[amask], (10 ** hi84)[amask], color=CNO_COLOR, alpha=0.16, zorder=2)
    ax.plot(r_ld[amask], (10 ** med)[amask], '-', color=CNO_COLOR, lw=2.6, zorder=3)

    # nitrogen points (one combined point per species) + power-law fit
    Nc = _combine_models(N)
    colors = {'N3': '#5B2B7B', 'N4': '#C2185B', 'N5': '#E67E22'}
    for _, p in Nc.iterrows():
        ax.errorbar(p['r'], p['Z'],
                    xerr=[[p['r'] - p['r_lo']], [p['r_hi'] - p['r']]],
                    yerr=[[p['Z'] - p['Z_lo']], [p['Z_hi'] - p['Z']]],
                    fmt='o', ms=10, color=colors[p['sp']],
                    mec='k', mew=0.7, ecolor=colors[p['sp']], elinewidth=2.2,
                    capsize=3, capthick=2.2, zorder=4)
    # legend handles: one per species (order N V, N IV], N III])
    from matplotlib.lines import Line2D
    sp_handles = [Line2D([0], [0], marker='o', color='w', mfc=colors[k], mec='k', ms=9, label=lbl)
                  for k, lbl in [('N5', r'${\rm N\,V}$'), ('N4', r'${\rm N\,IV]}$'), ('N3', r'${\rm N\,III]}$')]]

    # power-law fit to the 3 combined N points (same radial grid as the alpha curve)
    samp = _powerlaw_fit_N(Nc)             # shape (Nsamp, 3) = [m, b, lns]
    m_med = np.median(samp[:, 0])
    curves = samp[:, 1:2] + samp[:, 0:1] * np.log10(r_ld)[None, :]
    lo_c, mid_c, hi_c = np.percentile(curves, [16, 50, 84], axis=0)
    # draw the fit only from just left of the N V left error bar outward (no small-r extrapolation)
    fmask = (r_ld >= Nc.loc[Nc['sp'] == 'N5', 'r_lo'].iloc[0] * 10 ** -0.15) & (r_ld <= R_FIT_MAX)
    ax.fill_between(r_ld[fmask], (10 ** lo_c)[fmask], (10 ** hi_c)[fmask], color=N_COLOR, alpha=0.16, zorder=2)
    ax.plot(r_ld[fmask], (10 ** mid_c)[fmask], '-', color=N_COLOR, lw=2.4, zorder=3)

    ax.set_xscale('log'); ax.set_yscale('log')
    # x range follows the drawn data (N fit start, alpha curve extent, N error bars)
    xmin = min(r_ld[fmask].min(), r_ld[np.isfinite(med) & amask].min(), Nc['r_lo'].min())
    xmax = max(r_ld[np.isfinite(med) & amask].max(), Nc['r_hi'].max())
    ax.set_xlim(xmin * 10 ** -0.06, xmax * 10 ** 0.06)
    ax.set_ylim(0.02, 100)
    ax.set_xlabel(r'$r~[{\rm ld}]$', fontsize=22)
    ax.set_ylabel(r'$({\rm N/H})/({\rm N/H})_{\odot}$', color=N_COLOR, fontsize=22)
    ax2 = ax.twinx()
    ax2.set_yscale('log'); ax2.set_ylim(*ax.get_ylim())
    ax2.set_ylabel(r'$[{\rm (C+N+O)/H}]/[{\rm (C+N+O)/H}]_{\odot}$', rotation=270, labelpad=26, color=CNO_COLOR, fontsize=22)
    ax2.minorticks_on()
    ax2.tick_params(axis='y', which='major', length=10, width=1.5, direction='in', pad=5, labelsize=20)
    ax2.tick_params(axis='y', which='minor', length=5, width=1, direction='in')
    ax2.tick_params(axis='y', which='both', colors=CNO_COLOR)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True, length=10, width=1.5, direction='in', pad=5, labelsize=20)
    ax.tick_params(axis='both', which='minor', top=True, right=True, length=5, width=1, direction='in')
    ax.tick_params(axis='y', which='both', right=False, colors=N_COLOR)
    ax.spines['left'].set_color(N_COLOR);  ax2.spines['left'].set_color(N_COLOR)
    ax.spines['right'].set_color(CNO_COLOR);  ax2.spines['right'].set_color(CNO_COLOR)
    # top-right legends: species markers in one horizontal row, fit-slope lines below
    k_cno = np.median(df['k'].values)
    k_handles = [Line2D([0], [0], color=CNO_COLOR, lw=2.6, label=rf'$\langle k\rangle_{{\rm CNO}} = {k_cno:+.2f}$'),
                 Line2D([0], [0], color=N_COLOR, lw=2.4, label=rf'$k_{{\rm N}} = {m_med:+.2f}$')]
    leg_sp = ax.legend(handles=sp_handles, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                       ncol=3, columnspacing=1.2, handletextpad=0.4, fontsize=15, frameon=False)
    ax.add_artist(leg_sp)
    ax.legend(handles=k_handles, loc='upper right', bbox_to_anchor=(1.0, 0.90),
              fontsize=15, frameon=False)
    fig.savefig(out, dpi=200, bbox_inches='tight',
                bbox_extra_artists=[ax2.yaxis.label])
    plt.close(fig); print(f'Saved {out}')


def _drop_railed(df, rail_frac=0.05):
    """Drop objects whose posterior for k, log_r_ref, or log10_Z_norm piles up against a
    prior bound: the 16th percentile within `rail_frac` of the lower bound, or the 84th
    within `rail_frac` of the upper bound. Testing the percentiles (not just the median)
    catches posteriors railed against a bound whose median alone still clears the cut.
    p is intentionally NOT filtered: p = 0.5 (full breathing) is physical, so railing
    there is meaningful, not a sign of a bad fit."""
    bounds = {'k': (-1.0, 1.0), 'log_r_ref': (17.0, 19.5), 'log10_Z_norm': (-0.5, 1.30)}
    keep = np.ones(len(df), bool)
    for col, (lo, hi) in bounds.items():
        tol = rail_frac * (hi - lo)
        p16 = df[col].values - df[f'{col}_lo'].values
        p84 = df[col].values + df[f'{col}_hi'].values
        bad = (p16 < lo + tol) | (p84 > hi - tol)
        n = int(bad.sum())
        if n:
            print(f'  drop {n:3d} objects railing on {col} (16/84 pct within {tol:.3f} of [{lo}, {hi}])')
        keep &= ~bad
    return df[keep].reset_index(drop=True)


def main(csv=CSV, out_dir=OUT_DIR):
    df = pd.read_csv(csv).dropna(subset=['k', 'log_r_ref', 'log10_Z_norm', 'z']).reset_index(drop=True)
    df['logQ_med'] = _logQ_per_object(df['rm_id'].tolist()).values
    df['log_r_blr'] = df['log_r_ref'] + df['p'] * (df['logQ_med'] - LOG_Q_REF)
    print(f'{len(df)} objects loaded from {csv}')
    df = _drop_railed(df)
    print(f'{len(df)} objects after dropping railed fits')
    os.makedirs(out_dir, exist_ok=True)
    plot_Z_profiles_all(df,    f'{out_dir}/Z_profiles_all.png')
    plot_k_histogram(df,       f'{out_dir}/k_histogram.png')
    plot_k_vs_covariates(df,   f'{out_dir}/k_vs_covariates.png')
    plot_Z_profiles_population(df, f'{out_dir}/Z_profiles_population.png')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Population diagnostic plots from a LOC-fit summary CSV.')
    ap.add_argument('--tag', default='',
                    help='Read/write fits|plots/alpha/loc_gradient_noA_<tag>/ instead of the default _noA/.')
    args = ap.parse_args()
    suf = f'_{args.tag.lstrip("_")}' if args.tag else ''
    main(csv=f'fits/alpha/loc_gradient_noA{suf}/loc_fit_summary.csv',
         out_dir=f'plots/alpha/loc_gradient_noA{suf}')
