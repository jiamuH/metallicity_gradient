"""
Compute the nitrogen abundance (N/H)/(N/H)_sun implied by each nitrogen ion
(N III], N IV], N V) from the matched single-zone Cloudy reference runs, write
the result to a .dat file, and plot the inferred abundance vs BLR radius.

Two alpha-element-abundance models are used:
    m1  : Z_alpha = 1  Z_sun  (nagao_n10_phi19_N24_v100)
    m10 : Z_alpha = 10 Z_sun  (nagao_n10_phi19_N24_v100_m10)
Each run varies the nitrogen scale factor on a linear grid 0.1..9.9 (step 0.2).

Input:
    - Cloudy line-list outputs (matched runs, see MODEL_FILES below)
    - Observed line ratios from Van den Berk et al. (2001)

Output:
    - nitrogen/nitrogen_abundance_vs_r.dat
    - plots/nitrogen/Nitrogen_abundance_vs_rBLR.png

Usage (from repo root):
    python3 nitrogen/compute_nitrogen_abundance.py
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from plot_aox_line_ratios import parse_line_list  # reuse the line-list parser

plt.rcParams.update({
    'text.usetex': True,
    'axes.linewidth': 2,
    'font.family': 'serif',
    'font.weight': 'heavy',
    'font.size': 16,
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath} \boldmath'

# Custom colormap (same purple-pink-yellow palette as the notebooks / Fig. 2)
_CUSTOM_COLORS = [
    (0.15, 0.08, 0.40), (0.294, 0.161, 0.569), (0.529, 0.173, 0.635),
    (0.753, 0.212, 0.616), (0.918, 0.310, 0.533), (0.980, 0.471, 0.463),
    (0.965, 0.663, 0.478), (0.929, 0.851, 0.639),
]
CUSTOM_CMAP = LinearSegmentedColormap.from_list('custom_cmap', _CUSTOM_COLORS, N=256)
COLORS_CUSTOM = [CUSTOM_CMAP(x) for x in np.linspace(0.1, 0.9, 3)]  # N3, N4, N5

# ============================================================================
# Observed line fluxes (Van den Berk et al. 2001), same as Fig. 2
# ============================================================================
NV1240_obs   = 2.461;  NV1240_err   = 0.189
NIV1486_obs  = 0.258;  NIV1486_err  = 0.027
NIII1750_obs = 0.382;  NIII1750_err = 0.021
CIV1549_obs  = 25.29;  CIV1549_err  = 0.11

R_obs = {
    'N3': NIII1750_obs / CIV1549_obs,
    'N4': NIV1486_obs  / CIV1549_obs,
    'N5': NV1240_obs   / CIV1549_obs,
}
R_err = {
    'N3': R_obs['N3'] * np.hypot(NIII1750_err / NIII1750_obs, CIV1549_err / CIV1549_obs),
    'N4': R_obs['N4'] * np.hypot(NIV1486_err  / NIV1486_obs,  CIV1549_err / CIV1549_obs),
    'N5': R_obs['N5'] * np.hypot(NV1240_err   / NV1240_obs,   CIV1549_err / CIV1549_obs),
}

# ============================================================================
# Matched single-zone Cloudy nitrogen models (2 alpha-element abundances)
# ============================================================================
N_GRID = np.arange(0.1, 10.0, 0.2)  # nitrogen scale factor per grid block (50 values)

MODEL_FILES = {
    'm3':  ('/Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100_m3/'
            'strong_n10_phi19_N24_m3_v100_LineList_BLR_Fe2.txt'),
    'm10': ('/Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100_m10/'
            'strong_n10_phi19_N24_m10_v100_LineList_BLR_Fe2.txt'),
}
MODEL_LABEL = {'m3': r'$3\,Z_{\odot}$', 'm10': r'$10\,Z_{\odot}$'}
MODEL_MARKER = {'m3': 'o', 'm10': 's'}
MODEL_LS = {'m3': '-', 'm10': '--'}

LINE_COLS = {
    'N5': 'blnd 1240.00A',
    'N4': 'blnd 1486.00A',
    'N3': 'blnd 1750.00A',
    'C4': 'blnd 1549.00A',
}
LINE_DISPLAY = {'N3': r'$\rm N\,III]$', 'N4': r'$\rm N\,IV]$', 'N5': r'$\rm N\,V$'}
LINE_COLOR = {'N3': COLORS_CUSTOM[0], 'N4': COLORS_CUSTOM[1], 'N5': COLORS_CUSTOM[2]}

# N^{q+} -> N^{(q+1)+} ionization potentials [eV] for the species traced
ION_POTENTIAL = {'N3': 47.4, 'N4': 77.5, 'N5': 97.9}


def load_model_ratios(path):
    """Return dict {key: array over N grid} of line/CIV ratios for one model."""
    col_names, rows = parse_line_list(path)
    if rows.size == 0:
        raise RuntimeError(f'No data parsed from {path}')
    col_idx = {name: i for i, name in enumerate(col_names)}
    civ = rows[:, col_idx[LINE_COLS['C4']]]
    out = {}
    for key in ('N3', 'N4', 'N5'):
        out[key] = rows[:, col_idx[LINE_COLS[key]]] / civ
    return out


def find_best_N(R_mod, n_grid):
    """Interpolate model ratio vs N scale factor; return (best, low, high) dict
    for each nitrogen ion, where best/low/high correspond to the observed ratio
    and its +-1 sigma bounds.
    """
    Z_best, Z_low, Z_high = {}, {}, {}
    for key in ('N3', 'N4', 'N5'):
        ratio = R_mod[key]
        mask = np.isfinite(ratio)
        x, R = n_grid[mask], ratio[mask]
        order = np.argsort(R)
        f = interp1d(R[order], x[order], bounds_error=False, fill_value='extrapolate')
        obs, err = R_obs[key], R_err[key]
        Z_best[key] = float(f(obs))
        Z_low[key]  = float(f(obs - err))
        Z_high[key] = float(f(obs + err))
    return Z_best, Z_low, Z_high


# ============================================================================
# BLR radii: anchor N V to 2 light-days, scale by phi_opt assuming r ~ phi^-1/2
# ============================================================================
PHI_BEST  = {'N3': 17, 'N4': 19, 'N5': 21}
PHI_RANGE = {'N3': 0.5, 'N4': 1.5, 'N5': 1.5}
PHI_REF   = PHI_BEST['N5']  # anchor on N V
R_REF     = 2.0             # light-days


def blr_radii():
    r_blr, r_rng = {}, {}
    for line in ('N3', 'N4', 'N5'):
        r_blr[line] = R_REF * 10 ** (-0.5 * (PHI_BEST[line] - PHI_REF))
        dphi = PHI_RANGE[line]
        r_lo = R_REF * 10 ** (-0.5 * ((PHI_BEST[line] + dphi) - PHI_REF))
        r_hi = R_REF * 10 ** (-0.5 * ((PHI_BEST[line] - dphi) - PHI_REF))
        r_rng[line] = (r_lo, r_hi)
    return r_blr, r_rng


def powerlaw_fit(r, z):
    """Ordinary least-squares fit log z = a + alpha log r; return (alpha, a)."""
    lr, lz = np.log10(r), np.log10(z)
    alpha, a = np.polyfit(lr, lz, 1)
    return alpha, a


def fit_powerlaw_mcmc(log_r, log_z, sig_log_r, sig_log_z,
                      n_steps=120000, burn_frac=0.3, thin=8, seed=12345):
    """MCMC fit of a power law  log10 z = b + m log10 r  to data with errors in
    both coordinates plus an intrinsic scatter term.

    The (log-space) likelihood uses an effective variance
        var_i = sig_log_z_i^2 + m^2 sig_log_r_i^2 + s^2,
    where s (dex) is a free intrinsic-scatter parameter (Hogg, Bovy & Lang 2010).
    Flat priors:  m in (-2.0, 0.5),  b in (-3, 3),  ln s in (-12, 1.5).

    Returns an (N_samples, 3) array of posterior samples [m, b, ln s].
    """
    log_r = np.asarray(log_r, float)
    log_z = np.asarray(log_z, float)
    s2_z = np.asarray(sig_log_z, float) ** 2
    s2_r = np.asarray(sig_log_r, float) ** 2
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
        if lq - lp > np.log(rng.random()):
            p, lp = q, lq
        chain[i] = p
    return chain[int(burn_frac * n_steps)::thin]


def main():
    # ---- compute inferred N abundances ----
    results = {}
    for mkey, path in MODEL_FILES.items():
        R_mod = load_model_ratios(path)
        n_grid = N_GRID[: len(R_mod['N3'])]
        Zb, Zl, Zh = find_best_N(R_mod, n_grid)
        results[mkey] = {'Z_best': Zb, 'Z_low': Zl, 'Z_high': Zh}

    r_blr, r_rng = blr_radii()

    # ---- write .dat ----
    outfile = Path('nitrogen/nitrogen_abundance_vs_r.dat')
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, 'w') as f:
        f.write("# Nitrogen abundance (N/H)/(N/H)_sun vs BLR radius\n")
        f.write("# Matched single-zone Cloudy models (hden 10, phi 19, vturb 100, N24)\n")
        f.write("# + Van den Berk et al. (2001) observed line ratios\n")
        f.write("# Columns: species, r_blr [ld], r_low [ld], r_high [ld], "
                "model, Z_best, Z_low, Z_high\n")
        for line in ('N3', 'N4', 'N5'):
            r = r_blr[line]
            r_lo, r_hi = r_rng[line]
            for mkey in MODEL_FILES:
                Zb = results[mkey]['Z_best'][line]
                Zl = results[mkey]['Z_low'][line]
                Zh = results[mkey]['Z_high'][line]
                f.write(f"{line}  {r:.6f}  {r_lo:.6f}  {r_hi:.6f}  "
                        f"{mkey}  {Zb:.6f}  {Zl:.6f}  {Zh:.6f}\n")
    print(f"Saved {outfile}")

    print("\nBLR radii (light-days):")
    for line in ('N3', 'N4', 'N5'):
        r_lo, r_hi = r_rng[line]
        print(f"  {line}: r = {r_blr[line]:.2f}  [{r_lo:.2f}, {r_hi:.2f}]")
    print("\nInferred (N/H)/(N/H)_sun:")
    for mkey in MODEL_FILES:
        print(f"  {mkey}:")
        for line in ('N3', 'N4', 'N5'):
            Zb = results[mkey]['Z_best'][line]
            Zl = results[mkey]['Z_low'][line]
            Zh = results[mkey]['Z_high'][line]
            print(f"    {line}: {Zb:.3f}  [{Zl:.3f}, {Zh:.3f}]")

    # ---- MCMC power-law fit  log10(N/H) = b + m log10 r  to all (3 ions x 2 Z) points ----
    LINES = ('N3', 'N4', 'N5')
    log_r_pts, log_z_pts, sig_lr_pts, sig_lz_pts = [], [], [], []
    for mkey in MODEL_FILES:
        for line in LINES:
            r = r_blr[line]
            r_lo, r_hi = r_rng[line]
            zb = results[mkey]['Z_best'][line]
            zlo = min(results[mkey]['Z_low'][line], results[mkey]['Z_high'][line])
            zhi = max(results[mkey]['Z_low'][line], results[mkey]['Z_high'][line])
            log_r_pts.append(np.log10(r))
            log_z_pts.append(np.log10(zb))
            sig_lr_pts.append(0.5 * (np.log10(r_hi) - np.log10(r_lo)))
            sig_lz_pts.append(0.5 * (np.log10(zhi) - np.log10(zlo)))
    samples = fit_powerlaw_mcmc(log_r_pts, log_z_pts, sig_lr_pts, sig_lz_pts)
    m_s, b_s, lns_s = samples[:, 0], samples[:, 1], samples[:, 2]

    LN10 = np.log(10.0)
    g_s = m_s / LN10                                  # dlog10(N/H)/dr at r = 1 ld
    g_lo, g_c, g_hi = np.percentile(g_s, [16, 50, 84])
    m_lo, m_med, m_hi = np.percentile(m_s, [16, 50, 84])
    print(f"\nMCMC fit ({len(samples)} samples):  slope m = {m_med:.3f} "
          f"(+{m_hi - m_med:.3f}, -{m_med - m_lo:.3f}),  "
          f"intrinsic scatter s = {10 ** np.median(lns_s):.3f} dex")
    print(f"dlog(N/H)/dr|_1ld = {g_c:.3f}  (+{g_hi - g_c:.3f}, -{g_c - g_lo:.3f}) dex/ld")

    # ---- figure: inferred N abundance vs r_BLR ----
    fig, ax = plt.subplots(figsize=(7, 6))

    # span the full plotted x-range (set by the r_BLR error bars), with a small margin
    r_min_all = min(r_rng[l][0] for l in LINES)
    r_max_all = max(r_rng[l][1] for l in LINES)
    r_fit = np.logspace(np.log10(r_min_all) - 0.1, np.log10(r_max_all) + 0.1, 300)
    pred = b_s[:, None] + m_s[:, None] * np.log10(r_fit)[None, :]   # log10(N/H) samples
    z_lo, z_mid, z_hi = (10 ** np.percentile(pred, q, axis=0) for q in (16, 50, 84))
    # 68% posterior band of the power-law trend, with the posterior median as the central curve
    ax.fill_between(r_fit, z_lo, z_hi, color='0.6', alpha=0.30, zorder=1)
    ax.plot(r_fit, z_mid, color='0.35', lw=2.5, alpha=0.95, zorder=2)

    # data points: colour -> ion species, marker -> metallicity model
    for mkey in MODEL_FILES:
        zb = np.array([results[mkey]['Z_best'][l] for l in LINES])
        zl = np.array([results[mkey]['Z_low'][l] for l in LINES])
        zh = np.array([results[mkey]['Z_high'][l] for l in LINES])
        yerr_lo = np.clip(zb - np.minimum(zl, zh), 0, None)
        yerr_hi = np.clip(np.maximum(zl, zh) - zb, 0, None)
        for i, line in enumerate(LINES):
            r_lo, r_hi = r_rng[line]
            ax.errorbar(r_blr[line], zb[i],
                        xerr=[[r_blr[line] - r_lo], [r_hi - r_blr[line]]],
                        yerr=[[yerr_lo[i]], [yerr_hi[i]]],
                        fmt=MODEL_MARKER[mkey], ms=11, mfc=LINE_COLOR[line],
                        mec='k', ecolor=LINE_COLOR[line], elinewidth=1.8,
                        capsize=4, zorder=4)

    # gradient annotation box
    ann = (r'$\dfrac{d\log\,({\rm N/H})}{dr}\Big|_{1\,{\rm ld}} = '
           r'%.3f^{+%.3f}_{-%.3f}\,{\rm dex~ld^{-1}}$'
           % (g_c, g_hi - g_c, g_c - g_lo))
    ax.text(0.04, 0.93, ann, transform=ax.transAxes, fontsize=13,
            va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.4', lw=1.2))

    # legends: colour -> species (upper right), marker -> metallicity (lower left)
    species_handles = [Line2D([], [], marker='o', ls='', ms=11,
                              mfc=LINE_COLOR[k], mec='k') for k in LINES]
    species_labels = [LINE_DISPLAY[k] for k in LINES]
    model_handles = [Line2D([], [], marker=MODEL_MARKER[m], ls='', ms=11,
                            mfc='0.5', mec='k') for m in MODEL_FILES]
    model_labels = [MODEL_LABEL[m] for m in MODEL_FILES]
    leg1 = ax.legend(species_handles, species_labels, fontsize=15,
                     loc='upper right', frameon=False)
    ax.add_artist(leg1)
    ax.legend(model_handles, model_labels, fontsize=15,
              loc='lower left', frameon=False)

    ax.set_xscale('log')
    ax.set_xlabel(r'$r_{\rm BLR}~[\rm ld]$')
    ax.set_ylabel(r'$\rm (N/H)/(N/H)_{\odot}$')
    z_max = max(results[m]['Z_best'][l] for m in MODEL_FILES for l in LINES)
    ax.set_ylim(0, max(5.0, 1.2 * z_max))
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=False, right=True,
                   length=9, width=2, direction='in')
    ax.tick_params(axis='both', which='minor', top=False, right=True,
                   length=4, width=2, direction='in')

    # independent top axis: ionization potential of the traced N ions
    ax_top = ax.twiny()
    ax_top.set_xlim(min(ION_POTENTIAL.values()), max(ION_POTENTIAL.values()))
    ax_top.invert_xaxis()  # N V (highest I.P.) sits over the smallest r_BLR
    ax_top.set_xlabel(r'$\rm Ionization~Potential~[eV]$', labelpad=10)
    ax_top.minorticks_on()
    ax_top.tick_params(top=True, which='major', length=9, width=2, direction='in')
    ax_top.tick_params(top=True, which='minor', length=4, width=2, direction='in')

    out_png = Path('plots/nitrogen/Nitrogen_abundance_vs_rBLR.png')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == '__main__':
    main()
