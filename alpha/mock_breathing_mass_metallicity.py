"""
Population Monte Carlo: can partial breathing manufacture / steepen an
observed BLR-metallicity vs SMBH-mass relation?

Thought experiment made quantitative. Each AGN is caught at a random
luminosity state (breathing excursion Delta, in dex of L). Partial breathing
(radius responds as r ~ L^a with a < 1/2) does two correlated things:

  (1) biases the single-epoch virial mass, M_est ~ L^(0.5 - a) * FWHM-term,
      so log M_obs = log M_true + (0.5 - a) * Delta   (overestimate when bright)

  (2) raises the ionization parameter, U ~ L / r^2 ~ L^(1 - 2a), so the
      ionization-sensitive line ratio N V / C IV is boosted by S_U*(1-2a)*Delta,
      which an observer (calibrating at fixed U) reads as extra metallicity:
      log Z_inf = log Z_true + (S_U / A_Z) * (1 - 2a) * Delta

Because the SAME Delta drives M_obs up and Z_inf up, the population picks up a
spurious POSITIVE Z-M tilt on top of the true law  log Z_true = (2/15) log M_true.

Key quantities:
  a    : breathing partialness. a = 0.5 -> self-similar (U fixed, no effect);
         a = 0 -> frozen radius (maximal U swing).
  S_U  : d log(N V / C IV) / d log U   [PLACEHOLDER here; pin from LOC grid].
  A_Z  : d log(N V / C IV) / d log Z   (abundance sensitivity of the calibration).
  sigma_Delta : object-to-object spread in instantaneous log-luminosity state
                (single-epoch variability amplitude), in dex.
  sigma_logM  : true population spread in log M_BH, in dex.

Run:
    ~/miniconda3/envs/pypeit/bin/python alpha/mock_breathing_mass_metallicity.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from loc_gradient_fit import (load_cloudy_grid, make_interpolators,
                              model_line_ratios, LOGPHI_GRID, Z_GRID)

_LOC_INTERPS = None


def loc_interps():
    """Build (once) the trilinear interpolators over the Cloudy LOC grid."""
    global _LOC_INTERPS
    if _LOC_INTERPS is None:
        _LOC_INTERPS = make_interpolators(load_cloudy_grid())
    return _LOC_INTERPS

# ----------------------------------------------------------------------------
# Plot style (global rules)
# ----------------------------------------------------------------------------
plt.rcParams.update({'text.usetex': True, 'axes.linewidth': 2,
                     'font.family': 'serif', 'font.weight': 'heavy',
                     'font.size': 20})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

PLOT_DIR = "plots/alpha/breathing_mock"
os.makedirs(PLOT_DIR, exist_ok=True)


def style_axes(ax, major=8, minor=4):
    ax.minorticks_on()
    ax.tick_params(which='major', direction='in', length=major, width=1.5,
                   top=True, right=True)
    ax.tick_params(which='minor', direction='in', length=minor, width=1.0,
                   top=True, right=True)


# ----------------------------------------------------------------------------
# Core model
# ----------------------------------------------------------------------------
TRUE_SLOPE = 2.0 / 15.0  # theoretical Z propto M^(2/15)

# Photoionization sensitivities pinned from the Cloudy LOC grid
# (alpha/_extract_SU_AZ.py, operating point log U ~ -2, Z = 3 Zsun; abundances
# scaled linearly from solar).  NV/CIV is the fiducial; the alpha ratio flips
# the sign of S_U.
SU_GRID = {'NV/CIV': 0.135, '(SiIV+OIV)/CIV': -0.105}   # log U ~ -2
AZ_GRID = {'NV/CIV': 0.144, '(SiIV+OIV)/CIV': 0.140}
# high-ionization values (log U ~ -1), e.g. high-Eddington sources
SU_HIU = {'NV/CIV': 0.307, '(SiIV+OIV)/CIV': -0.015}
AZ_HIU = {'NV/CIV': 0.159, '(SiIV+OIV)/CIV': 0.310}
S_U_FID, A_Z_FID = SU_GRID['NV/CIV'], AZ_GRID['NV/CIV']


def observed_slope_analytic(true_slope, a, S_U, A_Z, var_logM, var_Delta):
    """Population regression slope of log Z_inf on log M_obs (closed form)."""
    cM = (0.5 - a)                  # log M_obs response to Delta
    cZ = (S_U / A_Z) * (1.0 - 2.0 * a)  # log Z_inf response to Delta
    cov = true_slope * var_logM + cZ * cM * var_Delta
    var = var_logM + cM ** 2 * var_Delta
    return cov / var


def observed_slope_lambda(g_true, a, S_U, A_Z, var_logLam, var_Delta):
    """Population regression slope of log Z_inf on log lambda_obs (closed form).

    log lambda_obs = log lambda_true + (0.5 + a) Delta   (L up x1, M up x(0.5-a)).
    g_true is the *intrinsic* Z-lambda slope (theory: ~0)."""
    cL = (0.5 + a)                       # log lambda_obs response to Delta
    cZ = (S_U / A_Z) * (1.0 - 2.0 * a)   # log Z_inf response to Delta
    cov = g_true * var_logLam + cZ * cL * var_Delta
    var = var_logLam + cL ** 2 * var_Delta
    return cov / var


def contamination_weights(a, sigma_Delta, sigma_logM, sigma_logLam):
    """Breathing share of the variance on the M and lambda axes."""
    vD = sigma_Delta ** 2
    wM = (0.5 - a) ** 2 * vD / (sigma_logM ** 2 + (0.5 - a) ** 2 * vD)
    wL = (0.5 + a) ** 2 * vD / (sigma_logLam ** 2 + (0.5 + a) ** 2 * vD)
    return wM, wL


def run_population(n=20000, a=0.25, S_U=S_U_FID, A_Z=A_Z_FID,
                   sigma_Delta=0.15, sigma_logM=0.4, sigma_logLam=0.30,
                   g_true=0.0, true_slope_M=TRUE_SLOPE,
                   sigma_meas=0.05, logM_mean=9.0, seed=0):
    """Draw a mock AGN population with breathing acting on both the virial mass
    and the single-epoch Eddington ratio. Returns a dict."""
    rng = np.random.default_rng(seed)

    logM_true = rng.normal(logM_mean, sigma_logM, n) if sigma_logM > 0 \
        else np.full(n, logM_mean)
    logLam_true = rng.normal(0.0, sigma_logLam, n) if sigma_logLam > 0 \
        else np.zeros(n)                              # centred true Eddington ratio
    # intrinsic metallicity: 2/15 on mass, g_true on Eddington ratio (theory: g=0)
    logZ_true = true_slope_M * (logM_true - logM_mean) + g_true * logLam_true

    Delta = rng.normal(0.0, sigma_Delta, n)           # luminosity-state excursion

    cM = (0.5 - a)
    cL = (0.5 + a)
    cZ = (S_U / A_Z) * (1.0 - 2.0 * a)

    logM_obs = logM_true + cM * Delta
    logLam_obs = logLam_true + cL * Delta
    logZ_inf = logZ_true + cZ * Delta + rng.normal(0.0, sigma_meas, n)

    return dict(
        logM_obs=logM_obs, logLam_obs=logLam_obs, logZ_inf=logZ_inf,
        logM_mean=logM_mean,
        slope_M=np.polyfit(logM_obs - logM_mean, logZ_inf, 1)[0],
        slope_Lam=np.polyfit(logLam_obs, logZ_inf, 1)[0],
    )


# ----------------------------------------------------------------------------
# Figure 1: scatter demonstration (fiducial population + single-mass limit)
# ----------------------------------------------------------------------------
def per_object_tilt(a, S_U, A_Z, axis='M'):
    """Breathing-only slope (single-value limit) = cZ / c_axis."""
    cZ = (S_U / A_Z) * (1.0 - 2.0 * a)
    c_axis = (0.5 - a) if axis == 'M' else (0.5 + a)
    return cZ / c_axis


def fig_scatter():
    """Population scatter: true law vs fitted, broad mass function (realistic
    dilution) and a narrow mass-bin (where breathing bites)."""
    fid = dict(a=0.25, S_U=S_U_FID, A_Z=A_Z_FID, sigma_Delta=0.15, sigma_meas=0.05)
    s_b = per_object_tilt(0.25, S_U_FID, A_Z_FID, 'M')

    r_b = run_population(sigma_logM=0.40, **fid)   # broad
    r_n = run_population(sigma_logM=0.12, **fid)   # narrow bin
    lMo_b, lZi_b, slope_b, m0 = r_b['logM_obs'], r_b['logZ_inf'], r_b['slope_M'], r_b['logM_mean']
    lMo_n, lZi_n, slope_n = r_n['logM_obs'], r_n['logZ_inf'], r_n['slope_M']

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    for ax, lMo, lZi, sl, sM, tag in (
            (axes[0], lMo_b, lZi_b, slope_b, 0.40, r'$\rm broad:~\sigma_{\log M}=0.40$'),
            (axes[1], lMo_n, lZi_n, slope_n, 0.12, r'$\rm narrow~bin:~\sigma_{\log M}=0.12$')):
        s = slice(None, None, 25)
        ax.scatter(lMo[s], lZi[s], s=7, color='0.6', alpha=0.35)
        xx = np.linspace(-1.3, 1.3, 50) + m0
        ax.plot(xx, TRUE_SLOPE * (xx - m0), 'k--', lw=2.5,
                label=r'$\rm true~(2/15=%.3f)$' % TRUE_SLOPE)
        ax.plot(xx, sl * (xx - m0), color='tab:blue', lw=2.5,
                label=r'$\rm fit~=%.3f$' % sl)
        ax.set_xlabel(r'$\log~M_{\rm BH}^{\rm obs}~[M_\odot]$')
        ax.set_xlim(m0 - 1.3, m0 + 1.3)
        ax.set_ylim(-0.45, 0.45)
        ax.legend(fontsize=13, loc='upper left', title=tag, title_fontsize=12)
        style_axes(ax)
    axes[0].set_ylabel(r'$\log~Z_{\rm inf}~[{\rm dex,~centred}]$')
    fig.suptitle(r'$\rm per\textrm{-}object~breathing~tilt~s_b=%.2f$' % s_b,
                 fontsize=16)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, 'mock_scatter.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  per-object tilt s_b = {s_b:.3f}  (single-mass limit)")
    print(f"  broad  (sig_logM=0.40): fit {slope_b:.4f}  vs true {TRUE_SLOPE:.4f}")
    print(f"  narrow (sig_logM=0.12): fit {slope_n:.4f}")
    print(f"  -> wrote {out}")


# ----------------------------------------------------------------------------
# Figure 2: how observed slope grows with breathing amplitude, for several a
# ----------------------------------------------------------------------------
def fig_slope_scan():
    sigma_Delta = np.linspace(0.0, 0.40, 60)
    A_Z, S_U, a = A_Z_FID, S_U_FID, 0.25
    s_b = per_object_tilt(a, S_U, A_Z)

    fig, ax = plt.subplots(figsize=(8, 7))
    # dilution depends on the true mass-function width: narrow bins let
    # breathing dominate, broad samples wash it out.
    cases = {0.10: 'tab:purple', 0.25: 'tab:blue', 0.40: 'tab:green'}
    for sM, c in cases.items():
        slope = observed_slope_analytic(TRUE_SLOPE, a, S_U, A_Z,
                                        sM ** 2, sigma_Delta ** 2)
        ax.plot(sigma_Delta, slope, color=c, lw=2.5,
                label=r'$\sigma_{\log M}=%.2f$' % sM)

    ax.axhline(TRUE_SLOPE, color='k', ls='--', lw=2, label=r'$\rm true~2/15$')
    ax.axhline(s_b, color='0.5', ls=':', lw=2,
               label=r'$\rm breathing~limit~s_b=%.1f$' % s_b)
    ax.set_xlabel(r'$\sigma_\Delta~\rm [dex~of~\log L~state]$')
    ax.set_ylabel(r'$\rm observed~slope~(d\log Z_{\rm inf}/d\log M_{\rm obs})$')
    ax.set_xlim(0, 0.40)
    ax.set_ylim(0.10, 0.55)
    ax.legend(fontsize=13, loc='upper left',
              title=r'$a=%.2f,~S_U=%.1f$' % (a, S_U), title_fontsize=12)
    style_axes(ax)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, 'mock_slope_scan.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> wrote {out}")


# ----------------------------------------------------------------------------
# Figure 3: what S_U / breathing combo is needed to reach a target slope
# ----------------------------------------------------------------------------
def fig_target_map():
    """Observed slope over (sigma_Delta, sigma_logM): which sample design lets
    breathing matter. Fixed a, S_U, A_Z (S_U to be pinned from the LOC grid)."""
    a, A_Z, S_U = 0.25, A_Z_FID, S_U_FID
    sig_arr = np.linspace(0.02, 0.40, 120)      # breathing amplitude
    sM_arr = np.linspace(0.05, 0.60, 120)       # true mass-function width
    SG, SM = np.meshgrid(sig_arr, sM_arr)
    slope = observed_slope_analytic(TRUE_SLOPE, a, S_U, A_Z, SM ** 2, SG ** 2)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.pcolormesh(SG, SM, slope, shading='auto', cmap='viridis')
    cs = ax.contour(SG, SM, slope, levels=[0.15, 0.20, 0.25, 0.30, 0.40],
                    colors='w', linewidths=1.8)
    ax.clabel(cs, fmt='%.2f', fontsize=12)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$\rm observed~slope$')
    ax.set_xlabel(r'$\sigma_\Delta~\rm [dex~of~\log L~state]$')
    ax.set_ylabel(r'$\sigma_{\log M}~\rm [dex,~true~mass~spread]$')
    style_axes(ax)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, 'mock_target_map.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  -> wrote {out}")


# ----------------------------------------------------------------------------
# Figure 4: breathing-induced Z-lambda_Edd slope (true g=0), vs amplitude
# ----------------------------------------------------------------------------
def fig_lambda_scan():
    """With NO intrinsic Z-lambda dependence (g=0), how much positive Z-lambda
    slope does breathing alone fake, for several intrinsic Eddington spreads."""
    sigma_Delta = np.linspace(0.0, 0.40, 60)
    A_Z, S_U, a, g = A_Z_FID, S_U_FID, 0.25, 0.0
    s_bL = per_object_tilt(a, S_U, A_Z, 'lambda')

    fig, ax = plt.subplots(figsize=(8, 7))
    cases = {0.15: 'tab:purple', 0.30: 'tab:blue', 0.50: 'tab:green'}
    for sL, c in cases.items():
        slope = observed_slope_lambda(g, a, S_U, A_Z, sL ** 2, sigma_Delta ** 2)
        ax.plot(sigma_Delta, slope, color=c, lw=2.5,
                label=r'$\sigma_{\log\lambda}=%.2f$' % sL)
    ax.axhline(0.0, color='k', ls='--', lw=2, label=r'$\rm true~(g=0)$')
    ax.axhline(s_bL, color='0.5', ls=':', lw=2,
               label=r'$\rm breathing~limit~s_b^\lambda=%.2f$' % s_bL)
    ax.set_xlabel(r'$\sigma_\Delta~\rm [dex~of~\log L~state]$')
    ax.set_ylabel(r'$\rm observed~slope~(d\log Z_{\rm inf}/d\log\lambda_{\rm obs})$')
    ax.set_xlim(0, 0.40)
    ax.set_ylim(-0.02, s_bL * 1.1)
    ax.legend(fontsize=13, loc='upper left',
              title=r'$a=%.2f,~S_U=%.1f$' % (a, S_U), title_fontsize=12)
    style_axes(ax)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, 'mock_lambda_scan.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  per-object tilt s_b(lambda) = {s_bL:.3f}")
    print(f"  -> wrote {out}")


# absolute log10 line ratios at z_norm=3 (from alpha/_extract_SU_AZ.py)
R0_GRID = {'NV/CIV': -1.702, '(SiIV+OIV)/CIV': -0.541}   # log U ~ -2
R0_HIU  = {'NV/CIV': -1.493, '(SiIV+OIV)/CIV': -0.601}   # log U ~ -1


def fig_scatter_M_lambda(key='NV/CIV', sigma_logM=0.50, sigma_logLam=0.30,
                         su=None, az=None, r0=0.0, true_slope_M=TRUE_SLOPE,
                         logM_mean=9.0, logLam_mean=-0.3, yquant='Zinf',
                         tag='mock_scatter_M_lambda'):
    """One grid-pinned population in both planes, with ACTUAL (un-centred) axes:
    log M_BH and log lambda_Edd on x; log Z_inf or the observed log(line ratio)
    on y.  r0 is the absolute log10(line ratio) zero-point (yquant='ratio')."""
    sU = SU_GRID[key] if su is None else su
    aZ = AZ_GRID[key] if az is None else az
    r = run_population(a=0.25, S_U=sU, A_Z=aZ, sigma_Delta=0.15,
                       sigma_logM=sigma_logM, sigma_logLam=sigma_logLam,
                       true_slope_M=true_slope_M, logM_mean=logM_mean,
                       sigma_meas=0.05)
    m0 = r['logM_mean']
    xM = r['logM_obs']                          # actual log M_BH
    xL = logLam_mean + r['logLam_obs']          # actual log lambda_Edd
    if yquant == 'ratio':
        lY = r0 + aZ * r['logZ_inf']            # actual log(line ratio)
        base, yfac = r0, aZ
        ylab = r'$\log~(\mathrm{%s})$' % key
    else:
        lY = r['logZ_inf']
        base, yfac = 0.0, 1.0
        ylab = r'$\log~Z_{\rm inf}~[{\rm dex,~centred}]$'
    sM_fit, bM = np.polyfit(xM, lY, 1)
    sL_fit, bL = np.polyfit(xL, lY, 1)
    s = slice(None, None, 25)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)

    # --- left: vs M_BH ---
    ax = axes[0]
    ax.scatter(xM[s], lY[s], s=7, color='0.6', alpha=0.35)
    xx = np.linspace(m0 - 1.4, m0 + 1.4, 50)
    tlab = r'$\rm true~(2/15)$' if true_slope_M else r'$\rm true~(no~M~dep.)$'
    ax.plot(xx, base + yfac * true_slope_M * (xx - m0), 'k--', lw=2.5, label=tlab)
    ax.plot(xx, sM_fit * xx + bM, color='tab:blue', lw=2.5,
            label=r'$\rm fit~slope=%.3f$' % sM_fit)
    ax.set_xlabel(r'$\log~M_{\rm BH}^{\rm obs}~[M_\odot]$')
    ax.set_ylabel(ylab)
    ax.set_xlim(m0 - 1.4, m0 + 1.4)
    ax.legend(fontsize=13, loc='upper left')

    # --- right: vs lambda_Edd ---
    ax = axes[1]
    ax.scatter(xL[s], lY[s], s=7, color='0.6', alpha=0.35)
    xl = np.linspace(logLam_mean - 1.0, logLam_mean + 1.0, 50)
    ax.plot(xl, base + 0.0 * xl, 'k--', lw=2.5, label=r'$\rm true~(g=0)$')
    ax.plot(xl, sL_fit * xl + bL, color='tab:red', lw=2.5,
            label=r'$\rm fit~slope=%.3f$' % sL_fit)
    ax.set_xlabel(r'$\log~\lambda_{\rm Edd}^{\rm obs}$')
    ax.set_xlim(logLam_mean - 1.0, logLam_mean + 1.0)
    ax.legend(fontsize=13, loc='upper left')

    yr = 0.55 if yquant != 'ratio' else 0.35
    for ax in axes:
        ax.set_ylim(base - yr, base + yr)
        style_axes(ax)
    label = {'NV/CIV': r'NV/CIV', '(SiIV+OIV)/CIV': r'(SiIV+OIV)/CIV'}[key]
    fig.suptitle(r'$\rm %s~~(s_b(M)=%.2f,~s_b(\lambda)=%.2f)$'
                 % (label, per_object_tilt(0.25, sU, aZ, 'M'),
                    per_object_tilt(0.25, sU, aZ, 'lambda')), fontsize=16)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f'{tag}.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  {key} [{yquant}]: fit slope(M)={sM_fit:.3f}, fit slope(lambda)={sL_fit:.3f}")
    print(f"  -> wrote {out}")


def scenarios():
    """How strong can the inferred trend get? Sweep the levers (NV/CIV).
    s_b(M) = 2 S_U/A_Z is fixed by S_U/A_Z; the population strength is set by the
    breathing weight w, boosted by small a, large sigma_Delta, narrow selection."""
    SF, AF = SU_GRID['NV/CIV'], AZ_GRID['NV/CIV']   # log U ~ -2
    SH, AH = SU_HIU['NV/CIV'], AZ_HIU['NV/CIV']      # log U ~ -1
    rows = [
        # name,                a,    sD,   sM,   sL,   S_U,  A_Z
        ("fiducial",          0.25, 0.15, 0.50, 0.30, SF, AF),
        ("weak breathing a=0.1", 0.10, 0.15, 0.50, 0.30, SF, AF),
        ("high variability",  0.25, 0.30, 0.50, 0.30, SF, AF),
        ("narrow selection",  0.25, 0.15, 0.20, 0.15, SF, AF),
        ("high ionization",   0.25, 0.15, 0.50, 0.30, SH, AH),
        ("high-ion + narrow", 0.25, 0.15, 0.20, 0.15, SH, AH),
        ("ALL combined",      0.10, 0.30, 0.20, 0.15, SH, AH),
    ]
    print("\n  --- ways to a stronger inferred trend (NV/CIV) ---")
    print(f"   {'scenario':>22} | {'a':>4} {'sD':>4} {'sM':>4} {'sL':>4} "
          f"{'S_U':>5} {'A_Z':>5} | {'obsM':>6} {'obsLam':>7}")
    print("   " + "-" * 78)
    for name, a, sD, sM, sL, SU, AZ in rows:
        obsM = observed_slope_analytic(TRUE_SLOPE, a, SU, AZ, sM**2, sD**2)
        obsL = observed_slope_lambda(0.0, a, SU, AZ, sL**2, sD**2)
        print(f"   {name:>22} | {a:4.2f} {sD:4.2f} {sM:4.2f} {sL:4.2f} "
              f"{SU:5.2f} {AZ:5.2f} | {obsM:6.3f} {obsL:7.3f}")
    print("   true: M=2/15=0.133, lambda=0;  s_b(M) ceiling = 2 S_U/A_Z")


def nv_civ_phi_curve(z_norm=3.0):
    """n-collapsed (simple mean over density) log10 line ratios vs log phi at
    fixed Z -- the LOC average, with phi the control variable (n marginalized)."""
    arr = load_cloudy_grid()
    coll = {k: np.mean(v, axis=0) for k, v in arr.items()}        # (phi, Z)
    j = int(np.argmin(np.abs(Z_GRID - z_norm)))
    return {'NV/CIV': np.log10(coll['N5'][:, j] / coll['C4'][:, j]),
            '(SiIV+OIV)/CIV': np.log10((coll['Si4'][:, j] + coll['O4'][:, j])
                                       / coll['C4'][:, j])}


def fig_scatter_phi(key='NV/CIV', p=0.25, logphi0=21.0, z_norm=3.0,
                    sigma_logM=0.50, sigma_logLam=0.20, sigma_Delta=0.15,
                    logM_mean=8.0, logLam_mean=-0.3, sigma_meas=0.0,
                    n=20000, seed=0, tag='mock_scatter_phi_NV'):
    """M_BH / lambda_Edd dependency with the line ratio taken from the
    n-collapsed LOC grid as a function of phi.  Breathing sets phi through the
    luminosity: phi = Q/(4 pi r^2), r ~ L^p, Q ~ L  =>  phi ~ L^(1-2p), so a
    luminosity excursion Delta gives d log phi = (1-2p) Delta.  The mean phi
    (=logphi0) is fixed across the population (R-L relation keeps phi constant
    with mean L); only breathing variability moves it.  Fixed Z (no abundance
    variation), so every M/lambda trend is the phi-breathing artifact."""
    curve = nv_civ_phi_curve(z_norm)[key]            # log10 ratio vs LOGPHI_GRID
    rng = np.random.default_rng(seed)
    logM_true = rng.normal(logM_mean, sigma_logM, n)
    logLam_true = rng.normal(logLam_mean, sigma_logLam, n)
    Delta = rng.normal(0.0, sigma_Delta, n)
    logphi_obs = logphi0 + (1.0 - 2.0 * p) * Delta   # breathing moves phi
    logM_obs = logM_true + (0.5 - p) * Delta         # single-epoch virial bias
    logLam_obs = logLam_true + (0.5 + p) * Delta
    lY = np.interp(logphi_obs, LOGPHI_GRID, curve) + rng.normal(0.0, sigma_meas, n)
    base_true = float(np.interp(logphi0, LOGPHI_GRID, curve))

    sM, bM = np.polyfit(logM_obs, lY, 1)
    sL, bL = np.polyfit(logLam_obs, lY, 1)
    s = slice(None, None, 25)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    ylab = r'$\log~(\mathrm{%s})~\rm [LOC~average]$' % key

    ax = axes[0]
    ax.scatter(logM_obs[s], lY[s], s=7, color='0.6', alpha=0.35)
    xx = np.linspace(logM_mean - 1.4, logM_mean + 1.4, 50)
    ax.plot(xx, base_true + 0 * xx, 'k--', lw=2.5, label=r'$\rm true~(no~M~dep.)$')
    ax.plot(xx, sM * xx + bM, color='tab:blue', lw=2.5,
            label=r'$\rm fit~slope=%.3f$' % sM)
    ax.set_xlabel(r'$\log~M_{\rm BH}^{\rm obs}~[M_\odot]$')
    ax.set_ylabel(ylab)
    ax.set_xlim(logM_mean - 1.4, logM_mean + 1.4)
    ax.legend(fontsize=13, loc='upper left')

    ax = axes[1]
    ax.scatter(logLam_obs[s], lY[s], s=7, color='0.6', alpha=0.35)
    xl = np.linspace(logLam_mean - 1.0, logLam_mean + 1.0, 50)
    ax.plot(xl, base_true + 0 * xl, 'k--', lw=2.5, label=r'$\rm true~(g=0)$')
    ax.plot(xl, sL * xl + bL, color='tab:red', lw=2.5,
            label=r'$\rm fit~slope=%.3f$' % sL)
    ax.set_xlabel(r'$\log~\lambda_{\rm Edd}^{\rm obs}$')
    ax.set_xlim(logLam_mean - 1.0, logLam_mean + 1.0)
    ax.legend(fontsize=13, loc='upper left')

    for ax in axes:
        ax.set_ylim(base_true - 0.35, base_true + 0.35)
        style_axes(ax)
    axes[0].text(0.96, 0.06, r'$p=%.2f,~\log\phi_0=%.1f,~Z=%.0f\,Z_\odot$'
                 % (p, logphi0, z_norm), transform=axes[0].transAxes,
                 ha='right', va='bottom', fontsize=14)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f'{tag}.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  phi-LOC {key}: <log R>={base_true:.3f}, fit slope(M)={sM:.3f}, "
          f"fit slope(lambda)={sL:.3f}")
    print(f"  -> wrote {out}")


def fig_RL_M_fwhm(key='NV/CIV', logphi0=21.0, z_norm=3.0, sigma_RL=0.30,
                  sigma_logM=0.50, sigma_logLam=0.30, logM_mean=8.0,
                  logLam_mean=-0.5, logFWHM_mean=3.55, n=20000, seed=0,
                  tag='mock_scatter_RL_M_fwhm'):
    """R-L-scatter population (no breathing, no EV1) shown as NV/CIV vs the
    observed M_BH (left) and vs CIV FWHM (right).  The line width is the virial
    velocity, FWHM ~ sqrt(M_true / R_actual), with R_actual = R-L(L) * 10^delta;
    using R-L(L) ~ L^0.5 and L ~ M_true * lambda_true:
        log FWHM = 0.25 logM_true - 0.25 log lambda_true - 0.5 delta (+ const).
    The same delta that sets phi (hence NV/CIV) sets the width, so the two are
    positively correlated -- and the M_BH trend is that same thing through the
    virial formula M_est ~ R-L * FWHM^2."""
    curve = nv_civ_phi_curve(z_norm)[key]
    rng = np.random.default_rng(seed)
    logM_true = rng.normal(logM_mean, sigma_logM, n)
    logLam_true = rng.normal(logLam_mean, sigma_logLam, n)
    delta = rng.normal(0.0, sigma_RL, n)

    logphi = logphi0 - 2.0 * delta
    lY = np.interp(logphi, LOGPHI_GRID, curve)
    logM_obs = logM_true - delta
    logFWHM = 0.25 * logM_true - 0.25 * logLam_true - 0.5 * delta
    logFWHM += logFWHM_mean - logFWHM.mean()         # zero-point -> ~5000 km/s
    base_true = float(np.interp(logphi0, LOGPHI_GRID, curve))

    logLam_obs = logLam_true + delta                 # inferred Eddington ratio
    FWHM = 10.0 ** logFWHM                            # linear, km/s
    sM, bM = np.polyfit(logM_obs, lY, 1)
    sMt, bMt = np.polyfit(logM_true, lY, 1)           # vs TRUE mass -> ~flat
    sF, bF = np.polyfit(FWHM, lY, 1)                  # log(ratio) vs linear FWHM
    s = slice(None, None, 25)
    vmin, vmax = np.percentile(logLam_obs, [1, 99])

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    ylab = r'$\log~(\mathrm{%s})~\rm [LOC~average]$' % key
    ckw = dict(c=logLam_obs[s], cmap='viridis', vmin=vmin, vmax=vmax,
               s=9, alpha=0.6)

    ax = axes[0]
    ax.scatter(logM_obs[s], lY[s], **ckw)
    xx = np.linspace(logM_mean - 1.6, logM_mean + 1.6, 50)
    ax.plot(xx, sM * xx + bM, color='k', lw=2.5,
            label=r'$\rm inferred~M_{\rm BH}:~d\log R/d\log M=%.3f$' % sM)
    ax.plot(xx, sMt * xx + bMt, color='k', ls='--', lw=2.5,
            label=r'$\rm true~M_{\rm BH}:~d\log R/d\log M=%.3f$' % sMt)
    ax.set_xlabel(r'$\log~M_{\rm BH}~[M_\odot]$')
    ax.set_ylabel(ylab)
    ax.set_xlim(logM_mean - 1.6, logM_mean + 1.6)
    ax.legend(fontsize=13, loc='upper left')

    ax = axes[1]
    sc = ax.scatter(FWHM[s], lY[s], **ckw)
    ax.set_xlabel(r'$\mathrm{FWHM}(\mathrm{CIV})~[{\rm km~s^{-1}}]$')
    ax.set_xlim(0.0, 8000.0)

    for ax in axes:
        ax.set_ylim(base_true - 0.6, base_true + 0.6)
        style_axes(ax)
    axes[0].text(0.96, 0.06, r'$\sigma_{R\text{-}L}=%.2f,~\log\phi_0=%.1f,~'
                 r'Z=%.0f\,Z_\odot$' % (sigma_RL, logphi0, z_norm),
                 transform=axes[0].transAxes, ha='right', va='bottom', fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.96, wspace=0.04)
    cax = fig.add_axes([0.895, 0.12, 0.015, 0.84])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(r'$\log~\lambda_{\rm Edd}^{\rm obs}$')
    out = os.path.join(PLOT_DIR, f'{tag}.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  RL-fwhm {key}: fit slope(M)={sM:.3f}, "
          f"fit slope(FWHM)={sF*1000:.3f} dex/(1000 km/s)")
    print(f"  -> wrote {out}")


def fig_scatter_RL(key='NV/CIV', logphi0=21.0, z_norm=3.0, sigma_RL=0.30,
                   sigma_logM=0.50, sigma_logLam=0.30, logM_mean=8.0,
                   logLam_mean=-0.5, n=20000, seed=0, tag='mock_scatter_RL_NV'):
    """Physically-built population, NO breathing, NO EV1.  Independent draws of
    M_BH and lambda_Edd; phi derived from the R-L relation with scatter delta:
        log R = R-L(L) + delta  =>  phi ~ L/R^2 ~ 10^(-2 delta)  (mean phi const).
    The SAME delta biases the single-epoch virial mass and Eddington ratio:
        log M_est = log M_true - delta,   log lambda_est = log lambda_true + delta.
    NV/CIV = LOC-average f(phi, Z) at fixed Z (no abundance variation).  So every
    trend comes from the shared R-L scatter delta."""
    curve = nv_civ_phi_curve(z_norm)[key]
    rng = np.random.default_rng(seed)
    logM_true = rng.normal(logM_mean, sigma_logM, n)
    logLam_true = rng.normal(logLam_mean, sigma_logLam, n)
    delta = rng.normal(0.0, sigma_RL, n)             # R-L scatter (dex in R)

    logphi = logphi0 - 2.0 * delta                   # compact BLR -> higher phi
    lY = np.interp(logphi, LOGPHI_GRID, curve)
    logM_obs = logM_true - delta                     # virial bias from R-L scatter
    logLam_obs = logLam_true + delta
    base_true = float(np.interp(logphi0, LOGPHI_GRID, curve))

    sM, bM = np.polyfit(logM_obs, lY, 1)
    sL, bL = np.polyfit(logLam_obs, lY, 1)
    s = slice(None, None, 25)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    ylab = r'$\log~(\mathrm{%s})~\rm [LOC~average]$' % key

    ax = axes[0]
    ax.scatter(logM_obs[s], lY[s], s=7, color='0.6', alpha=0.35)
    xx = np.linspace(logM_mean - 1.6, logM_mean + 1.6, 50)
    ax.plot(xx, base_true + 0 * xx, 'k--', lw=2.5, label=r'$\rm true~(no~M~dep.)$')
    ax.plot(xx, sM * xx + bM, color='tab:blue', lw=2.5,
            label=r'$\rm fit~slope=%.3f$' % sM)
    ax.set_xlabel(r'$\log~M_{\rm BH}^{\rm obs}~[M_\odot]$')
    ax.set_ylabel(ylab)
    ax.set_xlim(logM_mean - 1.6, logM_mean + 1.6)
    ax.legend(fontsize=13, loc='upper left')

    ax = axes[1]
    ax.scatter(logLam_obs[s], lY[s], s=7, color='0.6', alpha=0.35)
    xl = np.linspace(logLam_mean - 1.2, logLam_mean + 1.2, 50)
    ax.plot(xl, base_true + 0 * xl, 'k--', lw=2.5, label=r'$\rm true~(g=0)$')
    ax.plot(xl, sL * xl + bL, color='tab:red', lw=2.5,
            label=r'$\rm fit~slope=%.3f$' % sL)
    ax.set_xlabel(r'$\log~\lambda_{\rm Edd}^{\rm obs}$')
    ax.set_xlim(logLam_mean - 1.2, logLam_mean + 1.2)
    ax.legend(fontsize=13, loc='upper left')

    for ax in axes:
        ax.set_ylim(base_true - 0.6, base_true + 0.6)
        style_axes(ax)
    axes[0].text(0.96, 0.06, r'$\sigma_{R\text{-}L}=%.2f,~\log\phi_0=%.1f,~'
                 r'Z=%.0f\,Z_\odot$' % (sigma_RL, logphi0, z_norm),
                 transform=axes[0].transAxes, ha='right', va='bottom', fontsize=14)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f'{tag}.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  RL {key}: <log R>={np.mean(lY):.3f} (range {lY.min():.2f} to "
          f"{lY.max():.2f}), fit slope(M)={sM:.3f}, fit slope(lambda)={sL:.3f}")
    print(f"  -> wrote {out}")


def fig_scatter_LOC(key='NV/CIV', p=0.25, logQ0=55.0, z_norm=3.0,
                    log_r_ref=17.7, sigma_logM=0.50, sigma_logLam=0.20,
                    sigma_Delta=0.15, logM_mean=8.0, logLam_mean=-0.3,
                    sigma_meas=0.0, n=20000, seed=0, tag='mock_scatter_LOC_NV'):
    """Null case (no abundance variation: fixed z_norm, no gradient), with the
    line ratio computed SELF-CONSISTENTLY from the Cloudy LOC + partial-breathing
    model.  Each AGN at breathing state Delta has
        log(ratio) = log10 model_line_ratios(logQ0 + Delta, k=0, p, z_norm);
    the mean ionization is set by logQ0 (population R-L keeps U fixed across mean
    luminosity), and Delta swings U via partial breathing p.  Absolute level and
    breathing response both come from the grid -- no S_U/A_Z, no hand zero-point."""
    interps = loc_interps()
    # LOC line ratio as a function of the breathing excursion Delta
    dgrid = np.linspace(-1.3, 1.3, 131)
    rr, _, _ = model_line_ratios(interps, logQ0 + dgrid, k=0.0, p=p,
                                 log_r_ref=log_r_ref, z_norm=z_norm)
    logR_of_d = np.log10(rr[key])
    base_true = float(np.interp(0.0, dgrid, logR_of_d))   # no-breathing level

    rng = np.random.default_rng(seed)
    logM_true = rng.normal(logM_mean, sigma_logM, n)
    logLam_true = rng.normal(logLam_mean, sigma_logLam, n)
    Delta = rng.normal(0.0, sigma_Delta, n)
    logM_obs = logM_true + (0.5 - p) * Delta        # single-epoch virial bias
    logLam_obs = logLam_true + (0.5 + p) * Delta     # L up x1, M up x(0.5-p)
    lY = np.interp(Delta, dgrid, logR_of_d) + rng.normal(0.0, sigma_meas, n)

    sM, bM = np.polyfit(logM_obs, lY, 1)
    sL, bL = np.polyfit(logLam_obs, lY, 1)
    s = slice(None, None, 25)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    ylab = r'$\log~(\mathrm{%s})~\rm [LOC~model]$' % key

    ax = axes[0]
    ax.scatter(logM_obs[s], lY[s], s=7, color='0.6', alpha=0.35)
    xx = np.linspace(logM_mean - 1.4, logM_mean + 1.4, 50)
    ax.plot(xx, base_true + 0 * xx, 'k--', lw=2.5, label=r'$\rm true~(no~M~dep.)$')
    ax.plot(xx, sM * xx + bM, color='tab:blue', lw=2.5,
            label=r'$\rm fit~slope=%.3f$' % sM)
    ax.set_xlabel(r'$\log~M_{\rm BH}^{\rm obs}~[M_\odot]$')
    ax.set_ylabel(ylab)
    ax.set_xlim(logM_mean - 1.4, logM_mean + 1.4)
    ax.legend(fontsize=13, loc='upper left')

    ax = axes[1]
    ax.scatter(logLam_obs[s], lY[s], s=7, color='0.6', alpha=0.35)
    xl = np.linspace(logLam_mean - 1.0, logLam_mean + 1.0, 50)
    ax.plot(xl, base_true + 0 * xl, 'k--', lw=2.5, label=r'$\rm true~(g=0)$')
    ax.plot(xl, sL * xl + bL, color='tab:red', lw=2.5,
            label=r'$\rm fit~slope=%.3f$' % sL)
    ax.set_xlabel(r'$\log~\lambda_{\rm Edd}^{\rm obs}$')
    ax.set_xlim(logLam_mean - 1.0, logLam_mean + 1.0)
    ax.legend(fontsize=13, loc='upper left')

    for ax in axes:
        ax.set_ylim(base_true - 0.35, base_true + 0.35)
        style_axes(ax)
    # parameters as an in-axes annotation (no plot title, per plotting rules)
    axes[0].text(0.96, 0.06, r'$p=%.2f,~Z=%.0f\,Z_\odot$' % (p, z_norm),
                 transform=axes[0].transAxes, ha='right', va='bottom', fontsize=15)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, f'{tag}.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  LOC {key}: <log R>={base_true:.3f}, fit slope(M)={sM:.3f}, "
          f"fit slope(lambda)={sL:.3f}")
    print(f"  -> wrote {out}")


def compare_M_lambda():
    """For each grid-pinned line ratio: how much of the observed M and lambda
    correlation is breathing, and with what sign."""
    a, sD = 0.25, 0.15
    sM, sL = 0.50, 0.30
    wM, wL = contamination_weights(a, sD, sM, sL)
    print(f"\n  --- breathing contamination (a={a}, sig_D={sD}, "
          f"sig_logM={sM}, sig_logLam={sL}) ---")
    print(f"   weights:  w_M = {wM*100:.1f}%   w_lambda = {wL*100:.1f}%")
    print(f"   {'ratio':>16} | {'S_U':>6} {'A_Z':>6} | {'s_b(M)':>7} {'s_b(lam)':>8}"
          f" | {'obs M':>6} {'obs lam':>7}")
    print("   " + "-" * 70)
    for key in ('NV/CIV', '(SiIV+OIV)/CIV'):
        sU, aZ = SU_GRID[key], AZ_GRID[key]
        sbM = per_object_tilt(a, sU, aZ, 'M')
        sbL = per_object_tilt(a, sU, aZ, 'lambda')
        obsM = TRUE_SLOPE * (1 - wM) + sbM * wM       # true Z-M slope = 2/15
        obsL = 0.0 * (1 - wL) + sbL * wL              # true Z-lambda slope = 0
        print(f"   {key:>16} | {sU:6.3f} {aZ:6.3f} | {sbM:7.2f} {sbL:8.2f}"
              f" | {obsM:6.3f} {obsL:7.3f}")
    print("   true: M slope = 2/15 = 0.133,  lambda slope = 0")
    print("   -> breathing fakes OPPOSITE-sign trends in NV/CIV vs the alpha ratio")


if __name__ == '__main__':
    print(f"Fiducial (grid-pinned): a=0.25, S_U={S_U_FID}, A_Z={A_Z_FID}, "
          f"sigma_Delta=0.15, sigma_logM=0.4")
    # sanity: MC vs analytic at fiducial
    r = run_population()
    an = observed_slope_analytic(TRUE_SLOPE, 0.25, S_U_FID, A_Z_FID, 0.4**2, 0.15**2)
    anL = observed_slope_lambda(0.0, 0.25, S_U_FID, A_Z_FID, 0.30**2, 0.15**2)
    print(f"  MC slope(M)   = {r['slope_M']:.4f}   analytic = {an:.4f}  (should match)")
    print(f"  MC slope(lam) = {r['slope_Lam']:.4f}   analytic = {anL:.4f}  (should match)\n")
    fig_scatter()
    # fiducial ionization (log U ~ -2)
    fig_scatter_M_lambda('NV/CIV', tag='mock_scatter_M_lambda_NV')
    fig_scatter_M_lambda('(SiIV+OIV)/CIV', tag='mock_scatter_M_lambda_SiOIV')
    # high-ionization / high-Eddington subsample (log U ~ -1, narrower lambda range)
    print("high-ionization (log U ~ -1):")
    fig_scatter_M_lambda('NV/CIV', sigma_logLam=0.20,
                         su=SU_HIU['NV/CIV'], az=AZ_HIU['NV/CIV'],
                         tag='mock_scatter_M_lambda_NV_hiU')
    fig_scatter_M_lambda('(SiIV+OIV)/CIV', sigma_logLam=0.20,
                         su=SU_HIU['(SiIV+OIV)/CIV'], az=AZ_HIU['(SiIV+OIV)/CIV'],
                         tag='mock_scatter_M_lambda_SiOIV_hiU')
    # NULL: true Z independent of BOTH M and lambda -> every trend is breathing
    # (mass range 1e7-1e9, i.e. centred at logM=8, sigma=0.5 spans 7-9)
    print("null true relation (no M, no lambda dependence), M = 1e7-1e9:")
    fig_scatter_M_lambda('NV/CIV', sigma_logM=0.50, sigma_logLam=0.20,
                         true_slope_M=0.0, logM_mean=8.0,
                         su=SU_HIU['NV/CIV'], az=AZ_HIU['NV/CIV'],
                         tag='mock_scatter_M_lambda_NV_null')
    # same population, shown in the directly observed line-ratio plane,
    # with ACTUAL (un-centred) log(NV/CIV) and log lambda_Edd axes.
    # Zero-point anchored to the observed NV/CIV level (~ -0.5); the grid's
    # linear-N value (-1.49) is low because it lacks secondary nitrogen.
    # The breathing slope is independent of this zero-point.
    fig_scatter_M_lambda('NV/CIV', sigma_logM=0.50, sigma_logLam=0.20,
                         true_slope_M=0.0, logM_mean=8.0, logLam_mean=-0.3,
                         yquant='ratio', su=SU_HIU['NV/CIV'], az=AZ_HIU['NV/CIV'],
                         r0=-0.5,
                         tag='mock_scatter_M_lambda_NV_null_ratio')
    fig_slope_scan()
    fig_target_map()
    fig_lambda_scan()
    # self-consistent: line ratio straight from the LOC + breathing model
    print("LOC-model line ratio (self-consistent):")
    fig_scatter_LOC('NV/CIV', p=0.25, logQ0=55.0, tag='mock_scatter_LOC_NV')
    fig_scatter_LOC('(SiIV+OIV)/CIV', p=0.25, logQ0=55.0,
                    tag='mock_scatter_LOC_SiOIV')
    # frozen radius (p=0): full ionization swing; logQ0=56.5 keeps log U ~ -1
    fig_scatter_LOC('NV/CIV', p=0.0, logQ0=56.5, tag='mock_scatter_LOC_NV_p0')
    # phi-based LOC average (n-collapsed grid), breathing sets phi(L)
    print("phi-based LOC (n-collapsed grid, breathing phi ~ L^(1-2p)):")
    fig_scatter_phi('NV/CIV', p=0.25, logphi0=21.0, tag='mock_scatter_phi_NV')
    fig_scatter_phi('(SiIV+OIV)/CIV', p=0.25, logphi0=21.0,
                    tag='mock_scatter_phi_SiOIV')
    fig_scatter_phi('NV/CIV', p=0.0, logphi0=21.0, tag='mock_scatter_phi_NV_p0')
    # physically-built population: M_BH & lambda drawn, phi from R-L scatter,
    # no breathing, no EV1
    print("R-L-scatter population (no breathing, no EV1):")
    fig_scatter_RL('NV/CIV', tag='mock_scatter_RL_NV')
    fig_scatter_RL('(SiIV+OIV)/CIV', tag='mock_scatter_RL_SiOIV')
    fig_RL_M_fwhm('NV/CIV', tag='mock_scatter_RL_M_fwhm')
    compare_M_lambda()
    scenarios()
