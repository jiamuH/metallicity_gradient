"""Velocity-resolved broad-line profile demonstration for the metallicity-gradient paper.

Demonstration model (NOT a data analysis) for the planned appendix of
Diffusion_v1.2/metal_gradient.tex: how a radial alpha-element gradient
Z(r) = Z_norm (r/r_Q)^k would imprint on the velocity-resolved
(SiIV+OIV])/CIV ratio, under the idealized assumption of purely virial,
isotropic BLR kinematics (no outflow/inflow/turbulence; that caveat is
discussed in the appendix text).

Model:
  - Parameters: the RM249 best fit from the paper's integrated line-ratio MCMC
    (fits/alpha/loc_gradient_noA_Qref58/rm249_loc_bestfit.txt): gradient k,
    breathing p, log r_ref (BLR radius at Q_ref = 1e58 s^-1), Z_norm.
  - Single epoch: Q from the median observed F_1350 of RM249 via
    f1350_to_logQ.  The epoch's BLR radius is r_BLR(Q) = r_ref (Q/Q_ref)^p,
    and clouds at x = r/r_BLR see
      log phi(x,Q) = log phi_ref - 2 log x + (1-2p)(log Q - log Q_ref),
      log Z(x,Q)  = log Z_norm + k log x + p k (log Q - log Q_ref),
    the same conventions as loc_gradient_fit.model_line_ratios.
  - Radial shells x log-spaced over [0.1, 10] (the paper's LOC window).
    Each shell's line luminosity per dlog r uses the same LOC weighting:
    W propto x^(Gamma+3) n^(beta_n+1) with Gamma = beta_n = -1, the soft
    low-ionization (log U) taper, and phi/Z clipped to the Cloudy grid.
  - Kinematics: isotropic virial motion around M_bh = 1e8 Msun,
    v_circ(r) = sqrt(G M_bh / r), line-of-sight Gaussian kernel with
    sigma_los(r) = v_circ(r)/sqrt(3).
  - Profile: F_line(v) = sum over shells of dL/dlogx * Gaussian(v; sigma_los).
    Velocity-resolved ratio R(v) = F_SiIV+OIV(v) / F_CIV(v).

Overlay: the observed RM249 line profiles, taken from the PrepSpec mean
spectrum minus its continuum model (rm249_avg_w.dat, column 2 - column 4;
the PrepSpec BLR model component is available via
load_rm249_blr_profiles(component='blr')), converted to line-of-sight
velocity about the CIV 1549 and SiIV+OIV] 1400 blend centers.

Outputs (plots/alpha/velocity_resolved/):
  - velocity_profiles_bestfit.png        : normalized F(v) of CIV and SiIV+OIV]
                                           at the fitted k, with the observed
                                           RM249 profiles overplotted (log flux axis)
  - velocity_profiles_bestfit_linear.png : same, linear flux axis
  - velocity_profiles_bestfit_absflux.png: linear axis in absolute flux
                                           (F_lambda); the model is scaled by
                                           ONE factor, set by matching the CIV
                                           integrated line flux to RM249, so
                                           the model (SiIV+OIV])/CIV ratio is
                                           preserved
  - velocity_resolved_ratio.png          : R(v) for k = -0.5, 0, +0.5
                                           (other parameters at the best fit)

Run (from repo root, project env active):
    python3 -m alpha.velocity_resolved_profiles
"""
import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from alpha.loc_gradient_fit import (
    make_interpolators, _logu_weight, f1350_to_logQ,
    GAMMA_LOC, BETA_N_LOC, LOGPHI_LO_LOC, LOG_C_LIGHT,
    LOGPHI_GRID, Z_GRID, LOGQ_REF,
)

# --- plotting style (project convention; see ~/.claude/rules/plotting.md) ---
plt.rcParams.update({
    'text.usetex': True,
    'axes.linewidth': 2,
    'font.family': 'serif',
    'font.weight': 'heavy',
    'font.size': 20,
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

PLOT_DIR = 'plots/alpha/velocity_resolved'

# --- demonstration-model constants ---
BESTFIT_FILE = 'fits/alpha/loc_gradient_noA_Qref58/rm249_loc_bestfit.txt'
RATIO_FILE   = 'data/alpha/observed_line_ratio_data/rm249_line_ratios.dat'
RM249_Z      = 1.71678      # RM249 redshift (PrepSpec)
M_BH_MSUN    = 1e8          # black-hole mass [Msun] (assumed; sets the virial velocity scale)
K_VALUES     = (-0.5, 0.0, +0.5)   # gradient slopes shown in the ratio figure

G_CGS    = 6.674e-8         # gravitational constant [cm^3 g^-1 s^-2]
MSUN_G   = 1.989e33         # solar mass [g]
KMS      = 1e5              # km/s in cm/s
_LOG4PI  = np.log10(4.0 * np.pi)

# radial / density / velocity grids
LOGX_LO, LOGX_HI, N_X = -1.0, 1.0, 80     # x = r/r_Q window (paper convention)
LOGN_LO, LOGN_HI, N_N = 9.0, 12.0, 24     # cloud density window [cm^-3]
V_MAX_KMS, N_V        = 15000.0, 601      # line-of-sight velocity grid [km/s]
OBS_RATIO_BIN_KMS     = 1000.0            # bin width for the observed ratio [km/s]

# --- observed RM249 spectrum (PrepSpec mean-spectrum BLR component) ---
RM249_AVG_FILE = '/Users/jiamuh/sdssrm/rm249/rm249_avg_w.dat'
C_KMS          = 2.99792458e5             # speed of light [km/s]
LAM_C4_REST    = 1549.06                  # CIV doublet blend rest wavelength [A]
LAM_SI4O4_REST = 1400.0                   # SiIV+OIV] blend rest wavelength [A]


def load_rm249_blr_profiles(component='contsub'):
    """Observed RM249 broad-line profiles in velocity space.

    Reads the PrepSpec mean spectrum (restframe wavelength) and converts the
    windows around the CIV 1549 and SiIV+OIV] 1400 blends to line-of-sight
    velocity, v = c (lam - lam0)/lam0, limited to |v| < V_MAX_KMS.

    component = 'contsub': mean spectrum (column 2) minus the PrepSpec
    continuum model (column 4) -- data-driven line flux (NLR and Fe2 still
    included).  component = 'blr': the PrepSpec BLR model component
    (column 6).  The 1-sigma error is column 3 (error on the mean spectrum)
    in both cases.

    Returns {line: (v [km/s], F_lambda, sigma_F)} (fluxes in 1e-17
    erg/cm2/s/A) for 'C4', 'Si4O4'.
    """
    a = np.loadtxt(RM249_AVG_FILE)
    lam, err = a[:, 0], a[:, 2]
    flux = a[:, 1] - a[:, 3] if component == 'contsub' else a[:, 5]
    out = {}
    for key, lam0 in (('C4', LAM_C4_REST), ('Si4O4', LAM_SI4O4_REST)):
        v = C_KMS * (lam - lam0) / lam0
        m = np.abs(v) < V_MAX_KMS
        out[key] = (v[m], flux[m], err[m])
    return out


def load_bestfit(path=BESTFIT_FILE):
    """Read the RM249 best-fit parameters (k, p, log_r_ref, Z_norm) from the
    paper's integrated line-ratio MCMC output file."""
    vals = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or '=' not in line:
                continue
            name, rhs = line.split('=', 1)
            vals[name.strip()] = float(rhs.split('+/-')[0])
    return {'k': vals['k'], 'p': vals['p'], 'log_r_ref': vals['log_r_ref'],
            'z_norm': vals['Z_norm']}


def median_logq(path=RATIO_FILE, redshift=RM249_Z):
    """log10 Q of the median observed F_1350 epoch of RM249."""
    f1350 = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            f1350.append(float(line.split()[0]))
    return float(f1350_to_logQ(np.median(f1350), redshift))


def shell_luminosities(interps, k, p, log_r_ref, z_norm, logq):
    """dL/dlogx for each line on the shell grid, for gradient slope k.

    Same LOC weighting and breathing conventions as
    loc_gradient_fit.model_line_ratios, evaluated at the single epoch logq:
    W propto x^(Gamma+3) n^(beta_n+1), soft low-ionization taper, phi and Z
    clipped to the Cloudy grid.

    Returns (logx, dL, log_r_blr) with dL = {line: array(N_X)} for
    'C4', 'Si4', 'O4' and log_r_blr = log10 of the epoch's BLR radius [cm].
    """
    logx = np.linspace(LOGX_LO, LOGX_HI, N_X)
    logn = np.linspace(LOGN_LO, LOGN_HI, N_N)

    dlq = logq - LOGQ_REF
    log_r_blr = log_r_ref + p * dlq                                    # r_BLR(Q) [log cm]
    logphi_ref = LOGQ_REF - _LOG4PI - 2.0 * log_r_ref                  # flux at x=1, Q=Q_ref
    logphi = logphi_ref - 2.0 * logx + (1.0 - 2.0 * p) * dlq           # (N_X,)
    logZ = np.log10(z_norm) + k * logx + (p * k) * dlq                 # (N_X,)

    w_x = 10.0 ** (logx * (GAMMA_LOC + 3.0))
    w_n = 10.0 ** (logn * (BETA_N_LOC + 1.0))
    W = np.outer(w_n, w_x)                                             # (N_N, N_X)

    logU = logphi[None, :] - logn[:, None] - LOG_C_LIGHT
    W_eff = W * (logphi >= LOGPHI_LO_LOC)[None, :] * _logu_weight(logU)

    logphi_c = np.clip(logphi, LOGPHI_GRID.min(), LOGPHI_GRID.max())
    Z_c = np.clip(10.0 ** logZ, Z_GRID.min(), Z_GRID.max())
    n_clip_phi = int(np.sum(logphi != logphi_c))
    n_clip_Z = int(np.sum(10.0 ** logZ != Z_c))
    if n_clip_phi or n_clip_Z:
        print(f"  [shells] clipped to Cloudy grid: phi at {n_clip_phi}/{N_X} shells, "
              f"Z at {n_clip_Z}/{N_X} shells (k={k:+.2f})")

    LN, _ = np.meshgrid(logn, logx, indexing='ij')
    LPQ = np.broadcast_to(logphi_c, LN.shape)
    ZZ = np.broadcast_to(Z_c, LN.shape)
    qpts = np.stack([LN, LPQ, ZZ], axis=-1).reshape(-1, 3)

    dL = {key: np.sum(W_eff * interps[key](qpts).reshape(LN.shape), axis=0)
          for key in ('C4', 'Si4', 'O4')}
    return logx, dL, log_r_blr


def velocity_profiles(logx, dL, log_r_blr):
    """Project shell luminosities into line-of-sight velocity space.

    Isotropic virial motion: each shell at r = x r_BLR contributes a Gaussian
    velocity kernel with sigma_los(r) = sqrt(G M_bh / r) / sqrt(3).

    Returns (v [km/s], F) with F = {line: array(N_V)} for 'C4' and 'Si4O4'.
    """
    v = np.linspace(-V_MAX_KMS, V_MAX_KMS, N_V)                        # (N_V,)
    r_cm = 10.0 ** (log_r_blr + logx)                                  # (N_X,)
    v_circ = np.sqrt(G_CGS * M_BH_MSUN * MSUN_G / r_cm) / KMS          # km/s
    sigma = v_circ / np.sqrt(3.0)                                      # (N_X,)

    kern = np.exp(-0.5 * (v[None, :] / sigma[:, None]) ** 2) \
        / (np.sqrt(2.0 * np.pi) * sigma[:, None])                      # (N_X, N_V)
    dlogx = logx[1] - logx[0]

    F = {'C4': np.sum(dL['C4'][:, None] * kern, axis=0) * dlogx,
         'Si4O4': np.sum((dL['Si4'] + dL['O4'])[:, None] * kern, axis=0) * dlogx}
    return v, F


def _style_axis(ax):
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=8, width=1.5)
    ax.tick_params(which='minor', length=4, width=1)


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    fit = load_bestfit()
    logq = median_logq()
    log_r_blr = fit['log_r_ref'] + fit['p'] * (logq - LOGQ_REF)
    print(f"RM249 best fit: k = {fit['k']:+.3f}, p = {fit['p']:.3f}, "
          f"log r_ref = {fit['log_r_ref']:.3f}, Z_norm = {fit['z_norm']:.3f}")
    print(f"median epoch: log Q = {logq:.3f} -> log r_BLR = {log_r_blr:.3f}")
    print("Loading Cloudy grid and building interpolators ...")
    interps = make_interpolators()

    # --- profiles and ratios for the fitted k and the k sweep ---
    profiles = {}
    for k in (fit['k'],) + K_VALUES:
        logx, dL, lrb = shell_luminosities(interps, k, fit['p'],
                                           fit['log_r_ref'], fit['z_norm'], logq)
        v, F = velocity_profiles(logx, dL, lrb)
        profiles[k] = (v, F)
        r0 = F['Si4O4'][N_V // 2] / F['C4'][N_V // 2]
        print(f"  k = {k:+.2f}: line-center ratio R(0) = {r0:.3f}, "
              f"integrated ratio = {np.trapz(F['Si4O4'], v) / np.trapz(F['C4'], v):.3f}")

    # --- figure 1: normalized line profiles at the fitted k, with observed
    #     RM249, in a log-flux and a linear-flux version ---
    obs = load_rm249_blr_profiles()
    v, F = profiles[fit['k']]
    for yscale, fname in (('log', 'velocity_profiles_bestfit.png'),
                          ('linear', 'velocity_profiles_bestfit_linear.png')):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(v, F['C4'] / F['C4'].max(), lw=2.5, color='tab:blue',
                label=r'$\rm C\,IV~(model)$')
        ax.plot(v, F['Si4O4'] / F['Si4O4'].max(), lw=2.5, color='tab:red',
                label=r'$\rm Si\,IV+O\,IV]~(model)$')
        for key, color in (('C4', 'tab:blue'), ('Si4O4', 'tab:red')):
            v_o, f_o, e_o = obs[key]
            norm = f_o.max()
            ax.plot(v_o, f_o / norm, lw=1.2, drawstyle='steps-mid', color=color,
                    alpha=0.7,
                    label=(r'$\rm C\,IV~(RM249)$' if key == 'C4'
                           else r'$\rm Si\,IV+O\,IV]~(RM249)$'))
            ax.fill_between(v_o, (f_o - e_o) / norm, (f_o + e_o) / norm,
                            step='mid', color=color, alpha=0.2, lw=0)
        ax.set_xlabel(r'$v~[\rm km~s^{-1}]$')
        ax.set_ylabel(r'$F(v)~\rm [normalized]$')
        ax.set_xlim(-V_MAX_KMS, V_MAX_KMS)
        if yscale == 'log':
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 2.0)
        else:
            ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=14, frameon=False)
        _style_axis(ax)
        fig.tight_layout()
        out1 = os.path.join(PLOT_DIR, fname)
        fig.savefig(out1, dpi=200)
        plt.close(fig)
        print(f"saved {out1}")

    # --- figure 1b: absolute flux (F_lambda), linear axis.  The model has an
    #     arbitrary overall normalization; one common scale factor is fixed by
    #     matching the CIV integrated line flux to the observed one, so the
    #     model (SiIV+OIV])/CIV ratio is preserved in the plot. ---
    v_o, f_o, e_o = obs['C4']
    fint_obs_c4 = np.trapz(f_o, v_o) * LAM_C4_REST / C_KMS   # [1e-17 erg/cm2/s]
    dv = v[1] - v[0]
    scale = fint_obs_c4 / (F['C4'].sum() * dv)               # flux-per-km/s units
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, lam0, color, lab in (
            ('C4', LAM_C4_REST, 'tab:blue', r'$\rm C\,IV~(model)$'),
            ('Si4O4', LAM_SI4O4_REST, 'tab:red', r'$\rm Si\,IV+O\,IV]~(model)$')):
        ax.plot(v, scale * F[key] * C_KMS / lam0, lw=2.5, color=color, label=lab)
    for key, color, lab in (('C4', 'tab:blue', r'$\rm C\,IV~(RM249)$'),
                            ('Si4O4', 'tab:red', r'$\rm Si\,IV+O\,IV]~(RM249)$')):
        v_o, f_o, e_o = obs[key]
        ax.plot(v_o, f_o, lw=1.2, drawstyle='steps-mid', color=color,
                alpha=0.7, label=lab)
        ax.fill_between(v_o, f_o - e_o, f_o + e_o,
                        step='mid', color=color, alpha=0.2, lw=0)
    ax.set_xlabel(r'$v~[\rm km~s^{-1}]$')
    ax.set_ylabel(r'$F_\lambda~[\rm 10^{-17}~erg~s^{-1}~cm^{-2}~\AA^{-1}]$')
    ax.set_xlim(-V_MAX_KMS, V_MAX_KMS)
    ax.legend(fontsize=14, frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    out1b = os.path.join(PLOT_DIR, 'velocity_profiles_bestfit_absflux.png')
    fig.savefig(out1b, dpi=200)
    plt.close(fig)
    print(f"saved {out1b}")

    # --- observed velocity-resolved ratio: bin both continuum-subtracted
    #     profiles in velocity, ratio the binned fluxes, propagate errors;
    #     drop bins where the binned CIV flux is below 2 sigma. ---
    v_c4, f_c4, e_c4 = obs['C4']
    v_si, f_si, e_si = obs['Si4O4']
    edges = np.arange(-V_MAX_KMS, V_MAX_KMS + 1, OBS_RATIO_BIN_KMS)
    vb, rb, rb_err = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mc = (v_c4 >= lo) & (v_c4 < hi)
        ms = (v_si >= lo) & (v_si < hi)
        if not (mc.any() and ms.any()):
            continue
        m_c4 = f_c4[mc].mean()
        s_c4 = np.sqrt(np.sum(e_c4[mc] ** 2)) / mc.sum()
        m_si = f_si[ms].mean()
        s_si = np.sqrt(np.sum(e_si[ms] ** 2)) / ms.sum()
        if m_c4 < 2.0 * s_c4:
            continue
        r = m_si / m_c4
        vb.append(0.5 * (lo + hi))
        rb.append(r)
        rb_err.append(abs(r) * np.hypot(s_si / m_si, s_c4 / m_c4))

    # --- figure 2: velocity-resolved ratio for the k values ---
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {-0.5: 'tab:red', 0.0: 'k', +0.5: 'tab:blue'}
    for k in K_VALUES:
        v, F = profiles[k]
        ax.plot(v, F['Si4O4'] / F['C4'], lw=2.5, color=colors[k],
                label=rf'$k = {k:+.1f}$' if k != 0 else r'$k = 0$')
    ax.errorbar(vb, rb, yerr=rb_err, fmt='o', ms=5, color='gray',
                mfc='white', mew=1.5, elinewidth=1.5, capsize=3,
                label=r'$\rm RM249~(binned)$', zorder=1)
    ax.set_xlabel(r'$v~[\rm km~s^{-1}]$')
    ax.set_ylabel(r'$\rm (Si\,IV+O\,IV])/C\,IV$')
    ax.set_xlim(-V_MAX_KMS, V_MAX_KMS)
    ax.set_ylim(-0.3, 1.0)
    ax.legend(fontsize=14, frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    out2 = os.path.join(PLOT_DIR, 'velocity_resolved_ratio.png')
    fig.savefig(out2, dpi=200)
    plt.close(fig)
    print(f"saved {out2}")


if __name__ == '__main__':
    main()
