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
    logr_cm = np.linspace(15.5, 18.5, 200)
    r_ld = 10 ** logr_cm * LD_PER_CM
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    for _, row in df.iterrows():
        lrr, lZn, k = row['log_r_ref'], row['log10_Z_norm'], row['k']
        logZ = lZn + k * (logr_cm - lrr)
        mask = (logr_cm >= lrr + LOGX_LO) & (logr_cm <= lrr + LOGX_HI)
        ax.plot(r_ld[mask], 10 ** logZ[mask], '-', color=cmap(norm(k)), alpha=0.25, lw=1.2, zorder=2)
    ax.axhline(1.0, color='0.4', lw=1.0, ls=':', zorder=1)
    ax.axhline(20.0, color='0.4', lw=1.0, ls=':', zorder=1)
    medZ = np.full(logr_cm.size, np.nan)
    for i, lr in enumerate(logr_cm):
        vals = [row['log10_Z_norm'] + row['k'] * (lr - row['log_r_ref'])
                for _, row in df.iterrows()
                if row['log_r_ref'] + LOGX_LO <= lr <= row['log_r_ref'] + LOGX_HI]
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


def plot_Z_profiles_population(df, out):
    """Population-median Z(r) (0.1-dex bins) with nitrogen points + power-law fit overlaid.
    The alpha curve and the N fit share the same radial grid -- spanning from just below
    the smallest data error-bar foot (r_lo_min for N V) out to 1000 ld."""
    N = _load_nitrogen_points()
    r_lo_min = N['r_lo'].min()
    logr_cm = np.linspace(np.log10(r_lo_min / LD_PER_CM) - 0.05,
                          np.log10(1000.0 / LD_PER_CM), 51)            # ~0.07-dex bins, shared with the N fit
    r_ld = 10 ** logr_cm * LD_PER_CM
    # median is the extrapolation across all 205 power-laws (so the curve spans the full r range);
    # but the 16-84% band is restricted to where each object's LOC window covers (no extrapolation
    # uncertainty in the band) -- gives a wide curve with a band only where the fits constrain Z.
    lr_grid = logr_cm[None, :]
    lzn = df['log10_Z_norm'].values[:, None]
    k_v = df['k'].values[:, None]
    lrr = df['log_r_ref'].values[:, None]
    logZ_all = lzn + k_v * (lr_grid - lrr)                           # (n_obj, n_r); extrapolated
    med = np.percentile(logZ_all, 50, axis=0)
    loc_mask = (logr_cm[None, :] >= lrr + LOGX_LO) & (logr_cm[None, :] <= lrr + LOGX_HI)  # (n_obj, n_r)
    lo16 = np.full(logr_cm.size, np.nan); hi84 = np.full(logr_cm.size, np.nan)
    for i in range(logr_cm.size):
        vals = logZ_all[loc_mask[:, i], i]
        if vals.size >= 5:
            lo16[i], hi84[i] = np.percentile(vals, [16, 84])

    fig, ax = plt.subplots(figsize=(8, 6))
    # alpha-element population
    ax.fill_between(r_ld, 10 ** lo16, 10 ** hi84, color='steelblue', alpha=0.22, zorder=2)
    ax.plot(r_ld, 10 ** med, '-', color='steelblue', lw=2.6, zorder=3,
            label=rf'$\alpha~{{\rm elements~(population~median,}}~N={len(df)})$')

    # nitrogen points + power-law fit (N already loaded above for the shared grid)
    colors = {'N3': '#5B2B7B', 'N4': '#C2185B', 'N5': '#E67E22'}
    markers = {'m3': 'o', 'm10': 's'}
    for _, p in N.iterrows():
        ax.errorbar(p['r'], p['Z'],
                    xerr=[[p['r'] - p['r_lo']], [p['r_hi'] - p['r']]],
                    yerr=[[p['Z'] - p['Z_lo']], [p['Z_hi'] - p['Z']]],
                    fmt=markers[p['model']], ms=10, color=colors[p['sp']],
                    mec='k', mew=0.7, ecolor='0.35', elinewidth=1.0, capsize=2, zorder=4)
    # build legend handles for species + alpha-models
    from matplotlib.lines import Line2D
    sp_handles = [Line2D([0], [0], marker='o', color='w', mfc=colors[k], mec='k', ms=9, label=lbl)
                  for k, lbl in [('N3', r'${\rm N\,III]}$'), ('N4', r'${\rm N\,IV]}$'), ('N5', r'${\rm N\,V}$')]]
    am_handles = [Line2D([0], [0], marker=markers['m3'],  color='w', mfc='0.6', mec='k', ms=9, label=r'$Z_{\alpha} = 3\,Z_{\odot}$'),
                  Line2D([0], [0], marker=markers['m10'], color='w', mfc='0.6', mec='k', ms=9, label=r'$Z_{\alpha} = 10\,Z_{\odot}$')]

    # power-law fit to the N points (same radial grid as the alpha curve)
    samp = _powerlaw_fit_N(N)              # shape (Nsamp, 3) = [m, b, lns]
    m_med = np.median(samp[:, 0])
    curves = samp[:, 1:2] + samp[:, 0:1] * np.log10(r_ld)[None, :]
    lo_c, mid_c, hi_c = np.percentile(curves, [16, 50, 84], axis=0)
    n_fit_line = Line2D([0], [0], color='0.15', lw=2.4, label=rf'${{\rm N~power\,law:}}~\alpha={m_med:+.2f}$')
    ax.fill_between(r_ld, 10 ** lo_c, 10 ** hi_c, color='0.25', alpha=0.18, zorder=2)
    ax.plot(r_ld, 10 ** mid_c, '-', color='0.15', lw=2.4, zorder=4)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(0.1, 1000.0)
    ax.set_ylim(0.1, 30)
    ax.set_xlabel(r'$r~[{\rm ld}]$')
    ax.set_ylabel(r'$({\rm N/H})/({\rm N/H})_{\odot}$')
    ax2 = ax.twinx()
    ax2.set_yscale('log'); ax2.set_ylim(*ax.get_ylim())
    ax2.set_ylabel(r'$Z_{\rm C+N+O}/Z_{\odot}$', rotation=270, labelpad=22)
    ax2.tick_params(axis='y', which='major', length=8, width=1.5, direction='in', pad=5)
    ax2.tick_params(axis='y', which='minor', length=4, width=1, direction='in')
    _style(ax)
    leg1 = ax.legend(handles=sp_handles + am_handles + [n_fit_line], loc='lower left',
                     fontsize=11, frameon=False, ncol=2)
    ax.add_artist(leg1)
    ax.legend(loc='upper right', fontsize=11, frameon=False)
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig); print(f'Saved {out}')


def _drop_railed(df, rail_frac=0.05):
    """Drop objects whose posterior median for k, log_r_ref, or log10_Z_norm sits within
    `rail_frac` of either prior bound. p is intentionally NOT filtered: p = 0.5 (full
    breathing) is physical, so railing there is meaningful, not a sign of a bad fit."""
    bounds = {'k': (-1.0, 1.0), 'log_r_ref': (16.0, 18.5), 'log10_Z_norm': (-0.5, 1.30)}
    keep = np.ones(len(df), bool)
    for col, (lo, hi) in bounds.items():
        tol = rail_frac * (hi - lo)
        bad = (df[col].values < lo + tol) | (df[col].values > hi - tol)
        n = int(bad.sum())
        if n:
            print(f'  drop {n:3d} objects railing on {col} (within {tol:.3f} of [{lo}, {hi}])')
        keep &= ~bad
    return df[keep].reset_index(drop=True)


def main():
    df = pd.read_csv(CSV).dropna(subset=['k', 'log_r_ref', 'log10_Z_norm', 'z']).reset_index(drop=True)
    df['logQ_med'] = _logQ_per_object(df['rm_id'].tolist()).values
    print(f'{len(df)} objects loaded')
    df = _drop_railed(df)
    print(f'{len(df)} objects after dropping railed fits')
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_Z_profiles_all(df,    f'{OUT_DIR}/Z_profiles_all.png')
    plot_k_histogram(df,       f'{OUT_DIR}/k_histogram.png')
    plot_k_vs_covariates(df,   f'{OUT_DIR}/k_vs_covariates.png')
    plot_Z_profiles_population(df, f'{OUT_DIR}/Z_profiles_population.png')


if __name__ == '__main__':
    main()
