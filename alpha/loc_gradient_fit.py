"""
LOC + radial metallicity-gradient model with *partial* breathing, for the
broad-line-ratio vs continuum-luminosity relation in changing-look /
reverberation-mapped AGN.

Physical picture
----------------
- The BLR breathes, but not perfectly: r_BLR(Q) = r_BLR,ref * (Q/Q_ref)^p, with
  the breathing exponent p in [0, 0.5].  p = 0.5 is "perfect" R ~ L^1/2 (the BLR
  fully tracks the luminosity); p = 0 is "no breathing" (clouds fixed in cm);
  reality is in between (RM "breathing" measurements suggest p ~ 0.3-0.5).
- LOC clouds occupy a fixed *dimensionless* radius range x = r / r_BLR (and a
  fixed density range), with the standard Baldwin/Korista covering-fraction
  weighting (cover ∝ r^Gamma, density ∝ n^beta_n, plus the r^2 shell area).
- A cloud at dimensionless position x sees ionizing flux
      log phi(x, Q) = log phi_x  +  (1 - 2p) * (log Q - log Q_ref),
  where phi_x is its flux when Q = Q_ref.  So the *ionization* of every cloud
  changes with luminosity as Q^{1-2p}: full at p = 0 (the changing-look /
  ionization effect), zero at p = 0.5.  This is the effect that makes lines like
  C IV fade and reappear in CL-AGN -- it must be in the model.
- The metallicity gradient Z(r) = Z_norm * (r / r_1)^k is anchored in *physical*
  radius (it's a property of the gas, not the radiation field).  A cloud at
  fixed dimensionless position x is at r = x * r_BLR(Q) ∝ x * Q^p, so
      log Z(x, Q) = log(Z_norm) + k * log x  +  (p * k) * (log Q - log Q_ref),
  i.e. the gradient signal scales as p*k -- you only sample the gradient if the
  clouds actually move (p > 0).
- Per-cloud line emissivities I_line(n_H, phi, Z) come from the Cloudy LOC grid
  (strong_LOC_varym_N25).  phi and Z values outside the computed grid are clipped
  (explicit edge handling; note: the grid spans only Z >= 1 Z_sun and
  17 <= log phi <= 21, so weak breathing over a wide luminosity range will hit
  the phi edge -- harmless for the ~1-2 dex luminosity range of real CL/RM AGN).
- L_line(Q) = sum over the (phi, n) cloud grid of W(x,n) * I_line(phi(x,Q), n, Z(x,Q)).
  Line ratios = ratios of these.  The slope of ratio-vs-Q then carries BOTH an
  ionization term (∝ 1 - 2p) and a gradient term (∝ p*k); k = 0 does NOT give a
  flat relation in general -- it gives the pure ionization response.

This module provides:
  - the forward model (`model_line_ratios`, and the faster vectorized `model_ratios_at`);
  - demo plots of the predicted line ratios vs Q over ranges of k, p, log r_ref
    (`plot_model_demo`, `plot_model_2d`, `plot_model_vs_data`);
  - the MCMC fit (`fit_object`): free parameters k, p, log r_ref, log10(Z_norm),
    A_MgII, A_SiIV, log f (intrinsic scatter); C_Q is FIXED to the SED value.

Run (from repo root, project env active):
    python3 -m alpha.loc_gradient_fit                # forward-model demo plots
    python3 -m alpha.loc_gradient_fit --fit rm035    # MCMC fit for one object
"""
import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')                                       # file-only; safe for multiprocessing workers
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# --- plotting style (project convention; see ~/.claude/plotting.md) ---
plt.rcParams.update({
    'text.usetex': True,
    'axes.linewidth': 2,
    'font.family': 'serif',
    'font.weight': 'heavy',
    'font.size': 16,
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath} \boldmath'

# --- Cloudy LOC grid (per-cloud line intensities on a (log n_H, log phi, Z) grid) ---
CLOUDY_FILE          = '/Users/jiamuh/c23.01/my_models/loc_metal/strong_LOC_varym_N25_LineList_BLR_Fe2.txt'
CLOUDY_FILE_PHIEXT   = '/Users/jiamuh/c23.01/my_models/loc_metal/strong_LOC_varym_N25_phiext_LineList_BLR_Fe2.txt'
CLOUDY_FILE_SUBSOLAR = '/Users/jiamuh/c23.01/my_models/loc_metal/strong_LOC_varym_subsolar_N25_LineList_BLR_Fe2.txt'
# phi-axis: original [17,21] (9 vals) + extension [21.5,23] (4 vals) => [17,23] (13 vals).
# Z-axis : sub-solar [0.05, 0.95] step 0.1 (10 vals) + super-solar [1, 20] step 0.5 (39 vals) => 49 vals (non-uniform).
_LOGPHI_BASE = np.arange(17, 21.5, 0.5)
_LOGPHI_EXT  = np.arange(21.5, 23.1, 0.5)
_Z_SUBSOLAR  = np.round(np.arange(0.05, 1.0, 0.1), 3)    # 0.05, 0.15, ..., 0.95
_Z_SUPER     = np.arange(1.0, 20.5, 0.5)                 # 1.0, 1.5, ..., 20.0
LOGN_GRID   = np.arange(9, 12.5, 0.5)
LOGPHI_GRID = np.concatenate([_LOGPHI_BASE, _LOGPHI_EXT])
Z_GRID      = np.concatenate([_Z_SUBSOLAR, _Z_SUPER])
LINE_COLS = {'Mg2': 'blnd 2798.00A', 'C4': 'blnd 1549.00A',
             'Si4': 'BLND 1397.00A', 'O4': 'blnd 1402.00A',
             'N5': 'blnd 1240.00A'}

# --- model constants ---
Q_REF      = 1e58          # reference ionizing photon rate [s^-1]
LOGQ_REF   = np.log10(Q_REF)
LOGPHI_REF = 19.0          # ionizing flux at the fiducial position x = 1 when Q = Q_ref
GAMMA_LOC  = -1.0          # LOC radial covering-fraction slope (cover ∝ r^Gamma); BFKV95 use -1, range [-2,0]
BETA_N_LOC = -1.0          # LOC density-distribution slope (∝ n^beta_n)
LOGPHI_LO_LOC = -np.inf    # LOC integral excludes clouds with log phi < this (-inf = no floor)
LOGU_LO_LOC   = -3.0       # LOC low-ionization cut: clouds tapered out around this log U (CIV unreliable below it)
LOGU_TAPER_W  = 0.5        # full width (dex) of the soft taper centered on LOGU_LO_LOC
LOG_C_LIGHT   = np.log10(3e10)   # log10(c / cm s^-1); for the ionization parameter U = phi / (n_H c)
MGII_SCALE = 1.0           # fixed multiplicative weakening of the predicted MgII intensity (1.0 = no fudge)

# F1350 <-> Q conversion, FIXED from the assumed Nagao "strong bump" SED used in the
# Cloudy models:  C_Q = f_lambda(1350 A) / [ integral_{1 Ryd}^inf f_nu/(h nu) d nu ]
# (a pure SED-shape ratio, rest-frame; computed by compute_C_Q_nagao() below).
# This breaks the C_Q <-> r_ref degeneracy that the line-ratio data cannot resolve.
SED_CON_FILE   = '/Users/jiamuh/c23.01/my_models/loc_metal/strong_LOC_varym_N25_SED.con'
C_Q_NAGAO      = 1.366e-14     # erg cm^-2 s^-1 A^-1 per (photons cm^-2 s^-1)
LOG_C_Q_NAGAO  = np.log10(C_Q_NAGAO)   # ~ -13.864

_RYD_ERG = 13.605693 * 1.602176634e-12   # erg, energy of a 1-Rydberg photon
_NU_1350_RYD = 911.2671 / 1350.0         # photon energy at 1350 A, in Rydbergs (~0.6750)


def compute_C_Q_nagao(con_file=SED_CON_FILE):
    """Recompute C_Q from the Cloudy-saved incident continuum (.con) of the Nagao
    strong-bump SED.  C_Q = f_lambda(1350 A) / [ photon flux above 1 Ryd ] -- a pure
    SED-shape ratio (the overall normalization cancels).  Returns C_Q (erg cm^-2 s^-1
    A^-1 per photons cm^-2 s^-1).  Use to verify / refresh the C_Q_NAGAO constant."""
    nu, vfv = [], []
    with open(con_file) as fh:
        for line in fh:
            if 'GRID_DELIMIT' in line:        # delimiter line starts with '#': test before the skip
                break
            if line.startswith('#'):
                continue
            p = line.split('\t')
            try:
                nu.append(float(p[0])); vfv.append(float(p[1]))    # nu [Ryd], incident nu*f_nu [erg/cm2/s]
            except (ValueError, IndexError):
                continue
    nu = np.asarray(nu); vfv = np.asarray(vfv)
    fnu = np.where(nu > 0, vfv / np.maximum(nu, 1e-300), 0.0)       # f_nu = (nu f_nu)/nu (arb. norm)
    f_lambda_1350 = np.interp(_NU_1350_RYD, nu, fnu) * 911.2671 / 1350.0**2   # f_lambda = f_nu |dnu/dlambda|
    m = nu >= 1.0
    q_flux = np.trapezoid(fnu[m] / (nu[m] * _RYD_ERG), nu[m])      # photons cm^-2 s^-1 (arb. norm)
    return f_lambda_1350 / q_flux


# ---------------------------------------------------------------------------
def _load_one_cloudy_file(path, n_phi, n_z):
    """Read a single Cloudy LineList file into a {line: array(n_n, n_phi, n_z)} dict.
    File ordering: phi (outer), n_H (middle), Z (inner).  If the file is missing the
    very last data row (e.g. the most extreme grid corner failed to converge), pad
    that cell from its Z-neighbor.  If the file is missing its header row, we borrow
    the column names from CLOUDY_FILE (the canonical N25 line-list header)."""
    with open(path) as fh:
        first = fh.readline()
    has_header = first.startswith('#') or 'lineslist' in first.lower()
    if has_header:
        df = pd.read_csv(path, sep='\t', header=0)
    else:
        cols = pd.read_csv(CLOUDY_FILE, sep='\t', header=0, nrows=0).columns        # borrow header
        df = pd.read_csv(path, sep='\t', header=None, names=cols)
    df = df[~df.iloc[:, 0].astype(str).str.contains('GRID_DELIMIT')].reset_index(drop=True)
    nN = len(LOGN_GRID)
    expected = nN * n_phi * n_z
    if len(df) == expected - 1:
        print(f"  [load_cloudy_grid] {path.split('/')[-1]}: 1 row missing (last grid corner), padding from Z-neighbor")
        df = pd.concat([df, df.iloc[[-1]]], ignore_index=True)
    elif len(df) != expected:
        raise ValueError(f"Cloudy grid row count {len(df)} != {nN}*{n_phi}*{n_z} = {expected} in {path}")
    arr = {k: np.zeros((nN, n_phi, n_z)) for k in LINE_COLS}
    for j_phi in range(n_phi):
        for i_n in range(nN):
            idx = j_phi * nN + i_n
            sl = df.iloc[idx * n_z:(idx + 1) * n_z]
            for k, col in LINE_COLS.items():
                arr[k][i_n, j_phi, :] = sl[col].values
    return arr


def load_cloudy_grid(path=CLOUDY_FILE, path_ext=CLOUDY_FILE_PHIEXT, path_sub=CLOUDY_FILE_SUBSOLAR):
    """Load the original N25 (super-solar, log phi [17, 21]) grid plus the high-phi extension
    (log phi [21.5, 23], super-solar Z) and the sub-solar grid (Z [0.05, 0.95], full phi range
    [17, 23]).  Returns {line: array(len(LOGN_GRID), len(LOGPHI_GRID), len(Z_GRID))} on the
    unified (log n_H, log phi, Z) cube; Z axis is non-uniform (sub-solar step 0.1, super-solar 0.5)."""
    n_phi_base, n_phi_ext, n_phi_sub = len(_LOGPHI_BASE), len(_LOGPHI_EXT), len(LOGPHI_GRID)
    n_z_super, n_z_sub = len(_Z_SUPER), len(_Z_SUBSOLAR)
    # super-solar block: stitch base [17,21] and phiext [21.5,23] along the phi axis.
    super_blk = _load_one_cloudy_file(path, n_phi_base, n_z_super)
    if path_ext and os.path.exists(path_ext):
        ext = _load_one_cloudy_file(path_ext, n_phi_ext, n_z_super)
        super_blk = {k: np.concatenate([super_blk[k], ext[k]], axis=1) for k in LINE_COLS}  # (n_n, n_phi=13, n_z=39)
    # sub-solar block: already covers the full phi [17,23] in one file.
    if path_sub and os.path.exists(path_sub):
        sub_blk = _load_one_cloudy_file(path_sub, n_phi_sub, n_z_sub)                       # (n_n, 13, 10)
        return {k: np.concatenate([sub_blk[k], super_blk[k]], axis=2) for k in LINE_COLS}   # concat along Z
    return super_blk                                                                         # fallback: super-solar only


def make_interpolators(arr=None):
    """Trilinear interpolators over (log n_H, log phi, Z) for each line."""
    if arr is None:
        arr = load_cloudy_grid()
    pts = (LOGN_GRID, LOGPHI_GRID, Z_GRID)
    return {k: RegularGridInterpolator(pts, np.maximum(v, 1e-30),
                                       bounds_error=False, fill_value=None)
            for k, v in arr.items()}


# ---------------------------------------------------------------------------
_LOG4PI = np.log10(4.0 * np.pi)
# default r_BLR(Q_ref): the value that reproduces LOGPHI_REF ~ 19 (legacy choice)
LOG_R_REF_DEFAULT = (LOGQ_REF - _LOG4PI - LOGPHI_REF) / 2.0   # ~17.95  (log10 cm)
# value used for the demo plots: gives model line ratios near the observed ballpark
# (MgII/CIV ~ 0.6-0.7, with Z_norm ~ 2-3) -- note: below ~17 the cloud distribution
# starts running off the high-phi end of the Cloudy grid.
LOG_R_REF_PLOT = 17.7    # fiducial for fixed-r_ref demo panels (~ smallest r_ref where LOC window stays in extended phi grid)


def _logu_weight(logU):
    """Soft taper of the low-ionization cut: a smoothstep weight rising 0 -> 1
    across a LOGU_TAPER_W-dex ramp centered on LOGU_LO_LOC.  Replaces the hard
    log U > LOGU_LO_LOC mask so the LOC integral varies smoothly."""
    t = np.clip((logU - LOGU_LO_LOC) / LOGU_TAPER_W + 0.5, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def model_line_ratios(interps, logQ, k, p, log_r_ref=None, z_norm=3.0,
                      logx_lo=-1.0, logx_hi=1.0, n_x=80,
                      logn_lo=9.0, logn_hi=12.0, n_n=24, warn_clip=False):
    """Predicted broad-line ratios vs continuum level for gradient k, breathing p.

    Self-similar-with-partial-breathing LOC integral:
      - clouds occupy a fixed *dimensionless* radius range x = r/r_BLR ∈ [x_lo, x_hi]
        and a fixed density range, weighted by the LOC prescription
        W ∝ r^{Gamma+2} n^{beta_n}  (=> on the log x, log n grid, W ∝ x^{Gamma+3} n^{beta_n+1});
      - the fiducial BLR radius is r_BLR(Q_ref) = 10^{log_r_ref} cm, which breathes as
        r_BLR(Q) = r_BLR(Q_ref) (Q/Q_ref)^p;
      - a cloud at position x at luminosity Q sees
            log phi(x,Q) = LOGPHI_REF - 2 log x + (1-2p)(log Q - log Q_ref),
        where LOGPHI_REF = log Q_ref - log 4pi - 2 log_r_ref  (ionizing flux at x=1, Q=Q_ref);
      - its metallicity, with the gradient anchored at r_BLR(Q_ref), is
            log Z(x,Q) = log z_norm + k log x + p k (log Q - log Q_ref);
      - phi and Z are clipped to the Cloudy grid where they overshoot;
      - clouds in the low-ionization regime (log U = log phi - log n - log c
        near LOGU_LO_LOC) are soft-tapered out, where Cloudy CIV is unreliable.

    Parameters
    ----------
    interps : dict
        Output of `make_interpolators`.
    logQ : float or array
        log10 of the ionizing photon rate(s) (∝ continuum luminosity).
    k : float
        Metallicity-gradient slope: Z(r) ∝ r^k.
    p : float
        Breathing exponent: r_BLR(Q) ∝ Q^p (0 = clouds fixed in cm; 0.5 = perfect R∝L^{1/2}).
    log_r_ref : float or None
        log10(r_BLR(Q_ref) / cm).  Sets where the cloud distribution sits in
        ionizing-flux space.  None -> LOG_R_REF_DEFAULT (~17.95, i.e. LOGPHI_REF~19).
    z_norm : float
        Metallicity (Z/Z_sun) at x = 1 when Q = Q_ref (= at r = r_BLR(Q_ref)).
    logx_lo, logx_hi, n_x : float, float, int
        Cloud distribution in log10(x), x = r/r_BLR  (fixed shape).
    logn_lo, logn_hi, n_n : float, float, int
        Cloud distribution in log10(n_H).
    warn_clip : bool
        Print a warning when phi or Z is clipped to the model grid.

    Returns
    -------
    (ratios, logx, logphi_ref) where ratios is {'MgII/CIV': ..., '(SiIV+OIV)/CIV': ...}.
    """
    if log_r_ref is None:
        log_r_ref = LOG_R_REF_DEFAULT
    logQ = np.atleast_1d(np.asarray(logQ, dtype=float))
    logx = np.linspace(logx_lo, logx_hi, n_x)
    logn = np.linspace(logn_lo, logn_hi, n_n)
    logphi_ref = LOGQ_REF - _LOG4PI - 2.0 * log_r_ref          # ionizing flux at x=1, Q=Q_ref

    # LOC weight on the (uniform-log) x, n grid:  W ∝ x^{Gamma+3} * n^{beta_n+1}
    w_x = 10.0 ** (logx * (GAMMA_LOC + 3.0))                   # (n_x,)
    w_n = 10.0 ** (logn * (BETA_N_LOC + 1.0))                  # (n_n,)
    W = np.outer(w_n, w_x)                                     # (n_n, n_x)

    LN, _ = np.meshgrid(logn, logx, indexing='ij')             # (n_n, n_x)
    L = {key: np.zeros(logQ.size) for key in ('Mg2', 'C4', 'Si4', 'O4', 'N5')}
    n_clip_phi = n_clip_Z = 0
    for iq, lq in enumerate(logQ):
        dlq = lq - LOGQ_REF
        logphi_Q = logphi_ref - 2.0 * logx + (1.0 - 2.0 * p) * dlq        # (n_x,)
        logZ = np.log10(z_norm) + k * logx + (p * k) * dlq                # (n_x,)
        # drop clouds below the phi floor, and soft-taper clouds in the
        # low-ionization regime (around log U = LOGU_LO_LOC) where the Cloudy
        # CIV intensity is unreliably small.  log U uses the physical phi.
        logU = logphi_Q[None, :] - logn[:, None] - LOG_C_LIGHT            # (n_n, n_x)
        W_eff = W * (logphi_Q >= LOGPHI_LO_LOC)[None, :] * _logu_weight(logU)
        logphi_c = np.clip(logphi_Q, LOGPHI_GRID.min(), LOGPHI_GRID.max())
        Z_c = np.clip(10.0 ** logZ, Z_GRID.min(), Z_GRID.max())
        n_clip_phi += int(np.sum(logphi_Q != logphi_c))
        n_clip_Z += int(np.sum(10.0 ** logZ != Z_c))
        LPQ = np.broadcast_to(logphi_c, LN.shape)
        ZZ = np.broadcast_to(Z_c, LN.shape)
        qpts = np.stack([LN, LPQ, ZZ], axis=-1).reshape(-1, 3)
        if not W_eff.any():
            for key in L:
                L[key][iq] = np.nan
            continue
        for key in L:
            L[key][iq] = np.sum(W_eff * interps[key](qpts).reshape(LN.shape))

    if warn_clip and (n_clip_phi or n_clip_Z):
        print(f"  [model] clipped to grid: phi {n_clip_phi}, Z {n_clip_Z} "
              f"(k={k:+.2f}, p={p:.2f}, log_r_ref={log_r_ref:.2f}, logQ in [{logQ.min():.2f},{logQ.max():.2f}])")

    return ({'MgII/CIV': MGII_SCALE * L['Mg2'] / L['C4'],
             '(SiIV+OIV)/CIV': (L['Si4'] + L['O4']) / L['C4'],
             'NV/CIV': L['N5'] / L['C4']},
            logx, logphi_ref)


# ---------------------------------------------------------------------------
def plot_model_demo(z_norm=3.0, logQ_lo=54.5, logQ_hi=56.0, n_Q=40,
                    k_values=(-0.5, -0.25, 0.0, 0.25, 0.5), p_fixed=0.4,
                    p_values=(0.2, 0.3, 0.4, 0.5), k_fixed=0.0, log_r_ref=None,
                    out='plots/alpha/loc_gradient/model_ratio_vs_Q.png'):
    """Demo: line ratios vs continuum level.
    Top row: vary k at fixed p (= gradient effect on top of ionization).
    Bottom row: vary p at k = 0 (= pure ionization response, set by breathing efficiency)."""
    interps = make_interpolators()
    if log_r_ref is None:
        log_r_ref = LOG_R_REF_DEFAULT
    logQ = np.linspace(logQ_lo, logQ_hi, n_Q)
    keys = ['MgII/CIV', '(SiIV+OIV)/CIV']
    ylab = {'MgII/CIV': r'$\rm Mg\,II\,/\,C\,IV$',
            '(SiIV+OIV)/CIV': r'$\rm (Si\,IV + O\,IV)\,/\,C\,IV$'}
    xlab = r'$\log Q~[\rm photons~s^{-1}]$'

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True)

    # top row: vary k, p fixed
    cmap_k = plt.colormaps['plasma']
    norm_k = plt.Normalize(min(k_values), max(k_values))
    for k in k_values:
        rr, _, _ = model_line_ratios(interps, logQ, k, p_fixed, log_r_ref=log_r_ref,
                                     z_norm=z_norm, warn_clip=True)
        c = cmap_k(norm_k(k))
        for j, key in enumerate(keys):
            axes[0, j].plot(logQ, rr[key], '-', lw=3.0, color=c,
                            label=(rf'$k={k:+.2f}$' if j == 0 else None))
    for j, key in enumerate(keys):
        axes[0, j].set_ylabel(ylab[key])
    _lbb = dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7)
    axes[0, 0].text(0.04, 0.96, rf'$p={p_fixed:.2f},\ \log r_{{\rm ref}}={log_r_ref:.2f},\ Z_{{\rm norm}}={z_norm:g}\,Z_{{\odot}}$',
                    transform=axes[0, 0].transAxes, va='top', bbox=_lbb)
    axes[0, 0].legend(frameon=False, loc='upper right')
    sm_k = plt.cm.ScalarMappable(cmap=cmap_k, norm=norm_k); sm_k.set_array([])
    fig.colorbar(sm_k, ax=axes[0, :], pad=0.02, fraction=0.04).set_label(r'$k$', rotation=270, labelpad=18)

    # bottom row: vary p, k fixed (= 0 -> pure ionization)
    cmap_p = plt.colormaps['viridis']
    norm_p = plt.Normalize(min(p_values), max(p_values))
    for p in p_values:
        rr, _, _ = model_line_ratios(interps, logQ, k_fixed, p, log_r_ref=log_r_ref,
                                     z_norm=z_norm, warn_clip=True)
        c = cmap_p(norm_p(p))
        for j, key in enumerate(keys):
            axes[1, j].plot(logQ, rr[key], '-', lw=3.0, color=c,
                            label=(rf'$p={p:.2f}$' if j == 0 else None))
    for j, key in enumerate(keys):
        axes[1, j].set_ylabel(ylab[key])
        axes[1, j].set_xlabel(xlab)
    axes[1, 0].text(0.04, 0.96, rf'$k={k_fixed:.2f},\ \log r_{{\rm ref}}={log_r_ref:.2f},\ Z_{{\rm norm}}={z_norm:g}\,Z_{{\odot}}$',
                    transform=axes[1, 0].transAxes, va='top', bbox=_lbb)
    axes[1, 0].legend(frameon=False, loc='upper right')
    sm_p = plt.cm.ScalarMappable(cmap=cmap_p, norm=norm_p); sm_p.set_array([])
    fig.colorbar(sm_p, ax=axes[1, :], pad=0.02, fraction=0.04).set_label(r'$p$', rotation=270, labelpad=18)

    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            ax.set_yscale('log')
            ax.text(0.97, 0.04, rf'\textbf{{({chr(97 + i)}{j + 1})}}', transform=ax.transAxes,
                    va='bottom', ha='right')
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', top=True, right=True,
                           length=8, width=1.5, direction='in', pad=5)
            ax.tick_params(axis='both', which='minor', top=True, right=True,
                           length=4, width=1, direction='in')

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")
    # sanity prints
    rr_pp, _, _ = model_line_ratios(interps, logQ, 0.0, 0.5, z_norm=z_norm)   # k=0, perfect breathing -> flat
    print(f"  sanity: k=0, p=0.5 (perfect breathing) MgII/CIV log-range = "
          f"{np.ptp(np.log10(rr_pp['MgII/CIV'])):.4f} dex (should be ~0)")
    rr_p0, _, _ = model_line_ratios(interps, logQ, 0.0, 0.0, z_norm=z_norm)   # k=0, no breathing -> ionization slope
    print(f"  sanity: k=0, p=0.0 (no breathing)      MgII/CIV log-range = "
          f"{np.ptp(np.log10(rr_p0['MgII/CIV'])):.4f} dex (pure ionization; should be > 0)")


# ---------------------------------------------------------------------------
# 2D view of the fitting model:  predicted line ratio over (logQ, k) and (logQ, p)
# ---------------------------------------------------------------------------
_RM_OVERLAY = 'rm035'
_OBS_DIR = 'data/alpha/observed_line_ratio_data'
# per-object info for the data overlay:  redshift (for d_L), and fiducial per-ratio
# calibration offsets (borrowed from the windowed-model fit -- placement only).
# F1350 -> Q now uses the SED-fixed C_Q_NAGAO with f1350 taken to be rest-frame f_lambda.
_OBJ = {'rm035': dict(redshift=1.803090, offset_mg2=0.100, offset_si4=0.804)}


def f1350_to_logQ(f1350, redshift):
    """Convert rest-frame f_lambda(1350 A) [erg cm^-2 s^-1 A^-1] to log10 Q [s^-1]
    using the SED-fixed conversion:  Q = 4 pi d_L^2 f1350 / C_Q_NAGAO."""
    try:
        from astropy.cosmology import FlatLambdaCDM
        import astropy.units as u
        d_L_cm = FlatLambdaCDM(H0=70, Om0=0.3).luminosity_distance(redshift).to(u.cm).value
        log_4pi_dL2 = np.log10(4.0 * np.pi * d_L_cm ** 2)
    except Exception:
        log_4pi_dL2 = 58.364   # fallback for z~1.8, flat LCDM H0=70, Om0=0.3
    return np.log10(np.asarray(f1350)) + log_4pi_dL2 - LOG_C_Q_NAGAO


def _load_overlay_data(rm_id=_RM_OVERLAY):
    """Read observed (f1350, ratio) for the overlay object; convert f1350 -> logQ with
    the SED-fixed C_Q_NAGAO; divide ratios by the fiducial offsets (placement only).
    Returns dict {ratio_key: (logQ, ratio_corrected)} or None."""
    obj = _OBJ.get(rm_id)
    path = os.path.join(_OBS_DIR, f'{rm_id}_line_ratios.dat')
    if obj is None or not os.path.exists(path):
        return None
    f1350, ratio, rtype = [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split()
            if len(p) < 5:
                continue
            f1350.append(float(p[0])); ratio.append(float(p[1])); rtype.append(p[4])
    f1350 = np.array(f1350); ratio = np.array(ratio); rtype = np.array(rtype)
    logQ = f1350_to_logQ(f1350, obj['redshift'])
    out = {}
    m = (rtype == 'mg2_c4')
    if m.any():
        out['MgII/CIV'] = (logQ[m], ratio[m] / obj['offset_mg2'])
    m = (rtype == 'si4_c4')
    if m.any():
        out['(SiIV+OIV)/CIV'] = (logQ[m], ratio[m] / obj['offset_si4'])
    return out


# Observed line ratios: SDSS-RM sample mean +/- error of the mean.
# Per object the epoch ratios are averaged, then log10 taken; the sample
# mean and error of the mean (std/sqrt N) are over those per-object values.
# Stored as (mean of log10 ratio, error of the mean in dex):
#   MgII/CIV       : N=229  log = -0.389 +/- 0.016
#   (SiIV+OIV)/CIV : N=285  log = -0.567 +/- 0.017
OBS_RANGE = {
    'MgII/CIV':       (-0.3889, 0.0161),
    '(SiIV+OIV)/CIV': (-0.5666, 0.0171),
}


def plot_model_2d(z_norm=3.0, logQ_lo=53.5, logQ_hi=56.0, n_Q=70,
                  p_fixed=0.35, k_lo=-0.6, k_hi=0.6, n_k=60,
                  k_fixed=0.0, p_lo=0.0, p_hi=0.5, n_p=60,
                  logrref_lo=16.95, logrref_hi=17.95, n_r=60, logrref_fixed=None, p_fixed_c=None,
                  out='plots/alpha/loc_gradient/model_2d_ratio_maps.png'):
    """2D maps of the fitting model: predicted log(line ratio) over (logQ, k) at fixed
    (p, log_r_ref), over (logQ, p) at fixed (k, log_r_ref), and over (logQ, log_r_ref)
    at fixed (k, p).  Model only -- no data (k, p, log_r_ref are global, not per-epoch)."""
    interps = make_interpolators()
    if logrref_fixed is None:
        logrref_fixed = LOG_R_REF_PLOT
    if p_fixed_c is None:                  # p for the c-row (log_r_ref map); defaults to p_fixed
        p_fixed_c = p_fixed
    logQ = np.linspace(logQ_lo, logQ_hi, n_Q)
    k_grid = np.linspace(k_lo, k_hi, n_k)
    p_grid = np.linspace(p_lo, p_hi, n_p)
    r_grid = np.linspace(logrref_lo, logrref_hi, n_r)
    keys = ['MgII/CIV', '(SiIV+OIV)/CIV']
    klab = {'MgII/CIV': r'$\log(\rm Mg\,II\,/\,C\,IV)$',
            '(SiIV+OIV)/CIV': r'$\log[\rm (Si\,IV+O\,IV)\,/\,C\,IV]$'}
    xlab = r'$\log Q~[\rm photons~s^{-1}]$'

    def _stack(vary, **fixed):
        m = {key: np.zeros((len(vary[1]), n_Q)) for key in keys}
        for j, v in enumerate(vary[1]):
            kw = dict(fixed); kw[vary[0]] = v
            rr, _, _ = model_line_ratios(interps, logQ, kw['k'], kw['p'], log_r_ref=kw['log_r_ref'], z_norm=z_norm)
            for key in keys:
                m[key][j] = np.log10(rr[key])
        return m
    map_k = _stack(('k', k_grid), k=None, p=p_fixed, log_r_ref=logrref_fixed)
    map_p = _stack(('p', p_grid), k=k_fixed, p=None, log_r_ref=logrref_fixed)
    map_r = _stack(('log_r_ref', r_grid), k=k_fixed, p=p_fixed_c, log_r_ref=None)

    rows = [(k_grid, map_k, r'$k$', rf'$p={p_fixed:g},\ \log r_{{\rm ref}}={logrref_fixed:g},\ Z_{{\rm norm}}={z_norm:g}\,Z_{{\odot}}$'),
            (p_grid, map_p, r'$p$', rf'$k={k_fixed:g},\ \log r_{{\rm ref}}={logrref_fixed:g},\ Z_{{\rm norm}}={z_norm:g}\,Z_{{\odot}}$'),
            (r_grid, map_r, r'$\log r_{\rm ref}~[\rm cm]$', rf'$k={k_fixed:g},\ p={p_fixed_c:g},\ Z_{{\rm norm}}={z_norm:g}\,Z_{{\odot}}$')]

    fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True)
    fig.subplots_adjust(wspace=0.55, hspace=0.10)
    _bb = dict(boxstyle='round,pad=0.25', fc='black', ec='none', alpha=1.0)
    _lbox = dict(boxstyle='round,pad=0.3', fc='black', ec='none', alpha=1.0)
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    obs_proxy = Patch(facecolor='cyan', edgecolor='cyan', alpha=0.55, label=r'SDSS-RM mean')
    for irow, (ygrid, m, ylab, txt) in enumerate(rows):
        for j, key in enumerate(keys):
            ax = axes[irow, j]
            # colour range and contour levels adaptive to each panel's own data range
            vmn, vmx = float(np.nanmin(m[key])), float(np.nanmax(m[key]))
            im = ax.pcolormesh(logQ, ygrid, m[key], shading='auto', cmap='magma', vmin=vmn, vmax=vmx)
            levels = []
            if np.isfinite([vmn, vmx]).all() and vmx > vmn:
                levels = MaxNLocator(nbins=7, steps=[1, 2, 5, 10]).tick_values(vmn, vmx)
                levels = levels[(levels > vmn) & (levels < vmx)]
            if np.size(levels) and np.isfinite(m[key]).any():
                cs = ax.contour(logQ, ygrid, m[key], colors='w', linewidths=0.9, alpha=0.85, levels=levels)
                ax.clabel(cs, fmt=lambda v: rf'$\mathbf{{{v:.2f}}}$', fontsize=15, inline=True)
            # observed range: shade where the model log(ratio) lies within
            # the observed ratio band (center +/- half-width)
            obs, oerr = OBS_RANGE[key]
            if np.isfinite(m[key]).any():
                ax.contourf(logQ, ygrid, m[key], levels=[obs - oerr, obs + oerr],
                            colors=['cyan'], alpha=0.55, zorder=3)
                ax.contour(logQ, ygrid, m[key], levels=[obs - oerr, obs + oerr],
                           colors='cyan', linewidths=1.0, zorder=4)
            ax.legend(handles=[obs_proxy], loc='lower left', fontsize=11,
                      frameon=True, facecolor='0.12', edgecolor='none', labelcolor='w')
            fig.colorbar(im, ax=ax, pad=0.02, fraction=0.04).set_label(klab[key], rotation=270, labelpad=16)
            ax.set_ylabel(ylab)
            if irow == 2:
                ax.set_xlabel(xlab)
            ax.text(0.97, 0.04, rf'\textbf{{({chr(97 + irow)}{j + 1})}}', transform=ax.transAxes,
                    va='bottom', ha='right', color='w', fontsize=21, bbox=_lbox, zorder=20)
            ax.text(0.03, 0.96, txt, transform=ax.transAxes, va='top', ha='left', color='w',
                    bbox=_bb, zorder=20)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', top=True, right=True, length=8, width=1.5, direction='in', pad=5)
            ax.tick_params(axis='both', which='minor', top=True, right=True, length=4, width=1, direction='in')

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


def plot_model_vs_data(rm_id=_RM_OVERLAY, z_norm=3.0,
                       k_values=(-0.5, -0.25, 0.0, 0.25, 0.5), p_fixed=0.35,
                       p_values=(0.15, 0.25, 0.35, 0.45), k_fixed=0.0,
                       logrref_values=(17.4, 17.7, 18.0, 18.3), logrref_fixed=None,
                       out='plots/alpha/loc_gradient/model_vs_data_ratio_vs_Q.png'):
    """Model vs data: line ratio vs logQ (the space the data live in), with the
    object's epochs and model curves for small grids of (k, p, log_r_ref).
    Row 1: vary k at fixed p, log_r_ref.  Row 2: vary p at k=0, fixed log_r_ref.
    Row 3: vary log_r_ref at k=0, fixed p.  Data placed via the SED-fixed C_Q +
    fiducial calibration offsets (placement only, not a fit)."""
    interps = make_interpolators()
    obs = _load_overlay_data(rm_id)
    if logrref_fixed is None:
        logrref_fixed = LOG_R_REF_PLOT
    keys = ['MgII/CIV', '(SiIV+OIV)/CIV']
    ylab = {'MgII/CIV': r'$\rm Mg\,II\,/\,C\,IV$', '(SiIV+OIV)/CIV': r'$\rm (Si\,IV + O\,IV)\,/\,C\,IV$'}
    xlab = r'$\log Q~[\rm photons~s^{-1}]$'
    obscol = {'MgII/CIV': 'dodgerblue', '(SiIV+OIV)/CIV': 'deeppink'}

    if obs:
        allq = np.concatenate([v[0] for v in obs.values()])
        lq0, lq1 = allq.min() - 0.3, allq.max() + 0.3
    else:
        lq0, lq1 = 54.0, 56.0
    logQ = np.linspace(lq0, lq1, 60)

    fig, axes = plt.subplots(3, 2, figsize=(15, 16), sharex=True)

    cmap_k = plt.colormaps['plasma']; norm_k = plt.Normalize(min(k_values), max(k_values))
    for k in k_values:
        rr, _, _ = model_line_ratios(interps, logQ, k, p_fixed, log_r_ref=logrref_fixed,
                                     z_norm=z_norm, warn_clip=True)
        c = cmap_k(norm_k(k))
        for j, key in enumerate(keys):
            axes[0, j].plot(logQ, rr[key], '-', lw=2.8, color=c,
                            label=(rf'$k={k:+.2f}$' if j == 0 else None))
    cmap_p = plt.colormaps['viridis']; norm_p = plt.Normalize(min(p_values), max(p_values))
    for p in p_values:
        rr, _, _ = model_line_ratios(interps, logQ, k_fixed, p, log_r_ref=logrref_fixed,
                                     z_norm=z_norm, warn_clip=True)
        c = cmap_p(norm_p(p))
        for j, key in enumerate(keys):
            axes[1, j].plot(logQ, rr[key], '-', lw=2.8, color=c,
                            label=(rf'$p={p:.2f}$' if j == 0 else None))
    cmap_r = plt.colormaps['cividis']; norm_r = plt.Normalize(min(logrref_values), max(logrref_values))
    for lr in logrref_values:
        rr, _, _ = model_line_ratios(interps, logQ, k_fixed, p_fixed, log_r_ref=lr,
                                     z_norm=z_norm, warn_clip=True)
        c = cmap_r(norm_r(lr))
        for j, key in enumerate(keys):
            axes[2, j].plot(logQ, rr[key], '-', lw=2.8, color=c,
                            label=(rf'$\log r_{{\rm ref}}={lr:.1f}$' if j == 0 else None))

    for row in range(3):
        for j, key in enumerate(keys):
            ax = axes[row, j]
            if obs and key in obs:
                lqp, rp = obs[key]
                ax.plot(lqp, rp, 'o', ms=6, color=obscol[key], alpha=0.75, zorder=5,
                        label=(rf'$\rm {rm_id}\ data$' if j == 0 else None))
            ax.set_yscale('log')
            ax.set_ylabel(ylab[key])
            ax.text(0.97, 0.04, rf'\textbf{{({chr(97 + row)}{j + 1})}}', transform=ax.transAxes,
                    va='bottom', ha='right')
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', top=True, right=True, length=8, width=1.5, direction='in', pad=5)
            ax.tick_params(axis='both', which='minor', top=True, right=True, length=4, width=1, direction='in')
            if row == 2:
                ax.set_xlabel(xlab)
    _lbb = dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.7)
    axes[0, 0].text(0.04, 0.96, rf'$p={p_fixed:.2f},\ \log r_{{\rm ref}}={logrref_fixed:.2f},\ Z_{{\rm norm}}={z_norm:g}\,Z_{{\odot}}$',
                    transform=axes[0, 0].transAxes, va='top', bbox=_lbb)
    axes[1, 0].text(0.04, 0.96, rf'$k={k_fixed:.2f},\ \log r_{{\rm ref}}={logrref_fixed:.2f},\ Z_{{\rm norm}}={z_norm:g}\,Z_{{\odot}}$',
                    transform=axes[1, 0].transAxes, va='top', bbox=_lbb)
    axes[2, 0].text(0.04, 0.96, rf'$k={k_fixed:.2f},\ p={p_fixed:.2f},\ Z_{{\rm norm}}={z_norm:g}\,Z_{{\odot}}$',
                    transform=axes[2, 0].transAxes, va='top', bbox=_lbb)
    for r in range(3):
        axes[r, 0].legend(frameon=False, loc='best')
    s = plt.cm.ScalarMappable(cmap=cmap_k, norm=norm_k); s.set_array([])
    fig.colorbar(s, ax=axes[0, :], pad=0.02, fraction=0.04).set_label(r'$k$', rotation=270, labelpad=16)
    s = plt.cm.ScalarMappable(cmap=cmap_p, norm=norm_p); s.set_array([])
    fig.colorbar(s, ax=axes[1, :], pad=0.02, fraction=0.04).set_label(r'$p$', rotation=270, labelpad=16)
    s = plt.cm.ScalarMappable(cmap=cmap_r, norm=norm_r); s.set_array([])
    fig.colorbar(s, ax=axes[2, :], pad=0.02, fraction=0.04).set_label(r'$\log r_{\rm ref}$', rotation=270, labelpad=16)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


# ===========================================================================
# MCMC fit of the LOC + partial-breathing model to one object's epochs.
# Free parameters: k, p, log r_ref, log10(Z_norm), A_MgII, A_SiIV, log f.
# C_Q is FIXED to C_Q_NAGAO (so the C_Q <-> r_ref degeneracy is broken).
# ===========================================================================
_OBS_DATA_DIR = 'data/alpha/observed_line_ratio_data'

# Parameter list. With free per-line normalizations (default): 7 params
#   [k, p, log_r_ref, log10_Z_norm, A_MgII, A_SiIV, log_f]
# Without (use_free_a=False): 5 params [k, p, log_r_ref, log10_Z_norm, log_f]
# (the model then predicts the absolute ratio level; A_MgII = A_SiIV = 1).
_BASE_NAMES = ['k', 'p', 'log_r_ref', 'log10_Z_norm']
_A_NAMES = ['A_MgII', 'A_SiIV']
_TAIL_NAMES = ['log_f']
_NAME_LATEX = {'k': r'$k$', 'p': r'$p$',
               'log_r_ref': r'$\log r_{\rm ref}~[\rm cm]$',
               'log10_Z_norm': r'$\log Z_{\rm ref}~[Z_{\odot}]$',
               'A_MgII': r'$A_{\rm MgII}$',
               'A_SiIV': r'$A_{\rm SiIV}$', 'log_f': r'$\ln f$'}
# Half-width of the k prior and of the model grid in k.  Read from the environment so
# that multiprocessing workers (which re-import this module) see the same value; the
# --kmax option sets it.  Default 1.0 (the published fits).
KMAX = float(os.environ.get('LOC_KMAX', '1.0'))
PARAM_BOUNDS = {'k': (-KMAX, KMAX), 'p': (0.0, 0.5), 'log_r_ref': (17.0, 19.5),
                'log10_Z_norm': (-1.30, 1.30), 'A_MgII': (1e-3, 20.0),
                'A_SiIV': (1e-3, 20.0), 'log_f': (np.log(1e-3), np.log(2.0))}


def param_names(use_free_a=True):
    return _BASE_NAMES + (_A_NAMES if use_free_a else []) + _TAIL_NAMES


def param_latex(use_free_a=True):
    return [_NAME_LATEX[n] for n in param_names(use_free_a)]


PARAM_NAMES = param_names(True)        # default = free-A (kept for the grid build etc.)
PARAM_LATEX = param_latex(True)
_RCOL = {'mg2_c4': 0, 'si4_c4': 1}                                  # data ratio-type -> grid column
_RKEY = {'mg2_c4': 'MgII/CIV', 'si4_c4': '(SiIV+OIV)/CIV'}


def _names_for_ndim(ndim):
    return param_names(ndim >= len(_BASE_NAMES) + len(_A_NAMES) + len(_TAIL_NAMES))


def _bounds_arrays(names):
    return (np.array([PARAM_BOUNDS[n][0] for n in names]),
            np.array([PARAM_BOUNDS[n][1] for n in names]))


def _unpack_theta(theta):
    """theta -> (k, p, log_r_ref, log10_Z_norm, A_MgII, A_SiIV, log_f); fills A=1 if no free-A."""
    theta = np.asarray(theta, float)
    if len(theta) == 7:
        return tuple(theta)
    k, p, lrr, lZ, lf = theta
    return k, p, lrr, lZ, 1.0, 1.0, lf


def make_interpolator4(arr=None):
    """One RegularGridInterpolator over (log n_H, log phi, Z) whose value has a
    trailing length-4 axis = [Mg2, C4, Si4, O4] emissivities (one call -> all four)."""
    if arr is None:
        arr = load_cloudy_grid()
    stack = np.stack([np.maximum(arr[k], 1e-30) for k in ('Mg2', 'C4', 'Si4', 'O4')], axis=-1)
    return RegularGridInterpolator((LOGN_GRID, LOGPHI_GRID, Z_GRID), stack,
                                   bounds_error=False, fill_value=None)


def model_ratios_at(interp4, logQ, k, p, log_r_ref, z_norm,
                    logx_lo=-1.0, logx_hi=1.0, n_x=40, logn_lo=9.0, logn_hi=12.0, n_n=14):
    """Vectorized: predicted (MgII/CIV, (SiIV+OIV)/CIV) at an array of logQ for the
    LOC + partial-breathing model (same physics as model_line_ratios, faster)."""
    logQ = np.atleast_1d(np.asarray(logQ, float))
    logx = np.linspace(logx_lo, logx_hi, n_x)
    logn = np.linspace(logn_lo, logn_hi, n_n)
    logphi_ref = LOGQ_REF - _LOG4PI - 2.0 * log_r_ref
    W = np.outer(10.0 ** (logn * (BETA_N_LOC + 1.0)), 10.0 ** (logx * (GAMMA_LOC + 3.0)))   # (n_n, n_x)
    dlq = logQ - LOGQ_REF
    logphi_Q = logphi_ref - 2.0 * logx[None, :] + (1.0 - 2.0 * p) * dlq[:, None]            # (n_Q, n_x)
    logZ = np.log10(z_norm) + k * logx[None, :] + (p * k) * dlq[:, None]                    # (n_Q, n_x)
    valid = logphi_Q >= LOGPHI_LO_LOC                                                       # (n_Q, n_x): drop sub-floor clouds
    # soft-taper low-ionization clouds (around log U = LOGU_LO_LOC), where Cloudy
    # CIV is unreliably small; log U uses the physical (pre-clip) phi.
    logU = logphi_Q[:, None, :] - logn[None, :, None] - LOG_C_LIGHT                          # (n_Q, n_n, n_x)
    WQ = W[None, :, :] * valid[:, None, :] * _logu_weight(logU)                              # (n_Q, n_n, n_x)
    logphi_c = np.clip(logphi_Q, LOGPHI_GRID.min(), LOGPHI_GRID.max())
    Z_c = np.clip(10.0 ** logZ, Z_GRID.min(), Z_GRID.max())
    shp = (logQ.size, n_n, n_x)
    LN = np.broadcast_to(logn[None, :, None], shp)
    LPQ = np.broadcast_to(logphi_c[:, None, :], shp)
    ZZ = np.broadcast_to(Z_c[:, None, :], shp)
    I4 = interp4(np.stack([LN, LPQ, ZZ], axis=-1).reshape(-1, 3)).reshape(logQ.size, n_n, n_x, 4)
    L = np.einsum('Qnx,Qnxl->Ql', WQ, I4)          # (n_Q, 4) = [Mg2, C4, Si4, O4]; rows with no valid clouds -> 0
    with np.errstate(divide='ignore', invalid='ignore'):
        return MGII_SCALE * L[:, 0] / L[:, 1], (L[:, 2] + L[:, 3]) / L[:, 1]    # 0/0 -> NaN (rejected by the likelihood)


def load_object_data(rm_id, data_dir=_OBS_DATA_DIR, sigma_clip_n=3):
    """Read an object's line-ratio epochs.  Returns ({ratio_type: {logQ, R, sig}}, redshift),
    with F1350 -> logQ via the SED-fixed C_Q and a 3-sigma clip on the ratios."""
    import re
    from astropy.stats import sigma_clip
    path = os.path.join(data_dir, f'{rm_id}_line_ratios.dat')
    z = None
    rows = {'mg2_c4': [], 'si4_c4': []}
    with open(path) as fh:
        for line in fh:
            if line.startswith('#'):
                m = re.search(r'z\s*=\s*([\d.]+)', line)
                if m:
                    z = float(m.group(1))
                continue
            if not line.strip():
                continue
            p = line.split()
            if len(p) < 5:
                continue
            f1350, R, sig, t = float(p[0]), float(p[1]), float(p[2]), p[4]
            if t in rows and f1350 > 1e-19:
                rows[t].append((f1350, R, sig))
    if z is None:
        raise ValueError(f"No redshift found in header of {path}")
    out = {}
    for t, lst in rows.items():
        if not lst:
            continue
        a = np.array(lst, dtype=float)         # (N, 3): f1350, R, sigma
        if len(a) > 5:
            a = a[~sigma_clip(a[:, 1], sigma=sigma_clip_n, maxiters=3).mask]
        lq = f1350_to_logQ(a[:, 0], z)
        out[t] = dict(logQ=lq, R=a[:, 1], sig=a[:, 2], w=_density_weights(lq))
    return out, z


def _density_weights(logQ):
    """Weight each epoch by 1/sqrt(n in its logQ bin) so dense epoch clusters don't dominate
    (normalized so the effective sample size = N)."""
    n = len(logQ)
    nb = max(5, n // 10)
    edges = np.linspace(logQ.min() - 1e-9, logQ.max() + 1e-9, nb + 1)
    idx = np.clip(np.digitize(logQ, edges) - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    cnt[cnt == 0] = 1.0
    w = 1.0 / np.sqrt(cnt[idx])
    return w / w.sum() * n


# ---------------------------------------------------------------------------
# Precomputed model grid: tabulate the two predicted ratios over
# (log Q, k, p, log r_ref, log10 Z_norm), then interpolate it during the MCMC
# (~100x faster than re-running the LOC integral every likelihood call).
# ---------------------------------------------------------------------------
_GRID_LRR  = np.linspace(17.0, 19.5, 21)   # log_r_ref axis (step 0.125); set for Q_REF=1e58 (sensible radii for the new anchor)
_GRID_LOGQ = np.linspace(52.0, 57.0, 26)            # covers all objects' data ranges
_GRID_K    = np.linspace(-KMAX, KMAX, int(round(32 * KMAX)) + 1)   # step 0.0625
_GRID_P    = np.linspace(0.0, 0.5, 9)
_GRID_LZ   = np.linspace(-1.3, 1.3, 21)             # = log10(Z_norm/Z_sun); covers full Cloudy Z in [0.05, 20]
_LOC_LOGX_HALFWIDTH = 1.0                          # LOC window half-width in log r (mirrors the default in model_ratios_at)
LOC_GRID_FILE = (f'data/alpha/loc_gradient/loc_model_grid_gamma{GAMMA_LOC:+.1f}'
                 f'_mgsc{MGII_SCALE:.2f}_phimax{LOGPHI_GRID.max():.0f}'
                 f'_lrrmin{_GRID_LRR.min():.0f}'
                 f'_kmax{_GRID_K.max():.1f}_zmin{_GRID_LZ.min():+.1f}_zmax{_GRID_LZ.max():+.1f}'
                 f'_logxhw{_LOC_LOGX_HALFWIDTH:.2f}_logu{LOGU_LO_LOC:+.1f}t{LOGU_TAPER_W:.1f}.npz')
_GRID_NX, _GRID_NN = 24, 10                          # cloud-distribution sampling for the grid


def build_model_grid(out=None, n_x=_GRID_NX, n_n=_GRID_NN, verbose=True):
    """Tabulate (MgII/CIV, (SiIV+OIV)/CIV) over the (logQ, k, p, log_r_ref, log10_Z_norm)
    grid and save to `out` (.npz).  One-time cost, a couple of minutes."""
    out = out or LOC_GRID_FILE
    os.makedirs(os.path.dirname(out), exist_ok=True)
    interp4 = make_interpolator4()
    shp = (len(_GRID_LOGQ), len(_GRID_K), len(_GRID_P), len(_GRID_LRR), len(_GRID_LZ), 2)
    grid = np.empty(shp, dtype=np.float32)
    total = len(_GRID_K) * len(_GRID_P) * len(_GRID_LRR) * len(_GRID_LZ)
    c = 0
    for ik, k in enumerate(_GRID_K):
        for ip, p in enumerate(_GRID_P):
            for ir, lrr in enumerate(_GRID_LRR):
                for iz, lZ in enumerate(_GRID_LZ):
                    r_mg, r_si = model_ratios_at(interp4, _GRID_LOGQ, k, p, lrr, 10.0 ** lZ,
                                                 n_x=n_x, n_n=n_n)
                    grid[:, ik, ip, ir, iz, 0] = r_mg
                    grid[:, ik, ip, ir, iz, 1] = r_si
                    c += 1
                    if verbose and c % 2000 == 0:
                        print(f"  build_model_grid: {c}/{total}")
    np.savez_compressed(out, grid=grid, logQ=_GRID_LOGQ, k=_GRID_K, p=_GRID_P,
                        log_r_ref=_GRID_LRR, log10_Z_norm=_GRID_LZ)
    if verbose:
        print(f"  saved model grid {shp} to {out}")


def load_model_grid_interp(path=None):
    """Load the precomputed grid and return a RegularGridInterpolator over
    (logQ, k, p, log_r_ref, log10_Z_norm) with a trailing length-2 output
    [MgII/CIV, (SiIV+OIV)/CIV].  Builds the grid first if `path` is missing."""
    path = path or LOC_GRID_FILE
    if not os.path.exists(path):
        print(f"  model grid {path} not found -- building it now (one-time, ~2-3 min)...")
        build_model_grid(path)
    d = np.load(path)
    pts = (d['logQ'], d['k'], d['p'], d['log_r_ref'], d['log10_Z_norm'])
    return RegularGridInterpolator(pts, d['grid'].astype(float), bounds_error=False, fill_value=None)


def log_prior(theta):
    for name, v in zip(_names_for_ndim(len(theta)), theta):
        lo, hi = PARAM_BOUNDS[name]
        if not (lo < v < hi):
            return -np.inf
    return 0.0   # flat within the bounds


def log_likelihood(theta, data, grid_interp):
    """Per-epoch Gaussian likelihood in linear ratio space, with a fitted fractional
    intrinsic scatter f and epoch-density weighting.  Model ratios come from the
    precomputed grid via `grid_interp` (5D: logQ, k, p, log_r_ref, log10_Z_norm).
    `theta` may be 7-long (free A_MgII, A_SiIV) or 5-long (A's pinned to 1)."""
    k, p, lrr, lZ, A_mg, A_si, lf = _unpack_theta(theta)
    f = np.exp(lf)
    ll = 0.0
    for t, A, col in (('mg2_c4', A_mg, 0), ('si4_c4', A_si, 1)):
        if t not in data:
            continue
        d = data[t]
        N = len(d['logQ'])
        q = np.column_stack([d['logQ'], np.full(N, k), np.full(N, p),
                             np.full(N, lrr), np.full(N, lZ)])
        g = grid_interp(q)[:, col]
        m = A * g
        if not np.all(np.isfinite(m)) or np.any(m <= 0):
            return -np.inf
        var = d['sig'] ** 2 + (f * m) ** 2
        chi2 = (d['R'] - m) ** 2 / var
        ll += np.sum(d['w'] * (-0.5) * (chi2 + np.log(var)))
    return ll


def log_posterior(theta, data, grid_interp):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data, grid_interp)


_BOUNDS_CACHE = {len(param_names(b)): _bounds_arrays(param_names(b)) for b in (True, False)}


def log_posterior_vec(thetas, data, grid_interp):
    """Vectorized log-posterior: `thetas` is (n_walkers, ndim); returns (n_walkers,).
    For use with `emcee.EnsembleSampler(..., vectorize=True)`.  Handles ndim 7 or 5."""
    thetas = np.atleast_2d(thetas)
    nw, ndim = thetas.shape
    k, p, lrr, lZ = thetas[:, 0], thetas[:, 1], thetas[:, 2], thetas[:, 3]
    if ndim == 7:
        A = {'mg2_c4': thetas[:, 4], 'si4_c4': thetas[:, 5]}
        f = np.exp(thetas[:, 6])
    else:
        ones = np.ones(nw)
        A = {'mg2_c4': ones, 'si4_c4': ones}
        f = np.exp(thetas[:, 4])
    lo, hi = _BOUNDS_CACHE[ndim]
    out = np.full(nw, 0.0)
    out[np.any(thetas <= lo, axis=1) | np.any(thetas >= hi, axis=1)] = -np.inf
    live = np.isfinite(out)
    for t, col in (('mg2_c4', 0), ('si4_c4', 1)):
        if t not in data:
            continue
        d = data[t]
        N = len(d['logQ'])
        logQ = np.broadcast_to(d['logQ'], (nw, N))
        q = np.stack([logQ,
                      np.broadcast_to(k[:, None], (nw, N)),
                      np.broadcast_to(p[:, None], (nw, N)),
                      np.broadcast_to(lrr[:, None], (nw, N)),
                      np.broadcast_to(lZ[:, None], (nw, N))], axis=-1).reshape(-1, 5)
        g = grid_interp(q)[:, col].reshape(nw, N)
        m = A[t][:, None] * g
        bad = ~np.all(np.isfinite(m), axis=1) | np.any(m <= 0, axis=1)
        out[bad] = -np.inf
        live &= ~bad
        var = d['sig'][None, :] ** 2 + (f[:, None] * m) ** 2
        chi2 = (d['R'][None, :] - m) ** 2 / var
        ll = np.sum(d['w'][None, :] * (-0.5) * (chi2 + np.log(var)), axis=1)
        out[live] += ll[live]
    return out


def _plot_fit_result(rm_id, data, grid_interp, samples, out):
    tof = {'MgII/CIV': 'mg2_c4', '(SiIV+OIV)/CIV': 'si4_c4'}
    keys = [k for k in ['MgII/CIV', '(SiIV+OIV)/CIV'] if tof[k] in data]
    Acol = {'MgII/CIV': 4, '(SiIV+OIV)/CIV': 5}
    scol = {'MgII/CIV': 0, '(SiIV+OIV)/CIV': 1}
    ylab = {'MgII/CIV': r'$\rm Mg\,II\,/\,C\,IV$', '(SiIV+OIV)/CIV': r'$\rm (Si\,IV + O\,IV)\,/\,C\,IV$'}
    col = {'MgII/CIV': 'dodgerblue', '(SiIV+OIV)/CIV': 'deeppink'}
    allq = np.concatenate([d['logQ'] for d in data.values()])
    lq = np.linspace(allq.min() - 0.1, allq.max() + 0.1, 50)
    nA = samples.shape[1] == 7                                # free per-line normalizations?
    rng = np.random.default_rng(0)
    draws = samples[rng.choice(len(samples), size=min(400, len(samples)), replace=False)]
    nd, nq = len(draws), lq.size
    med = np.median(samples, axis=0)
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    k_over = np.linspace(-1.0, 1.0, 11)                       # 11 fixed-k overlays, every 0.2 dex
    k_norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    k_cm   = plt.cm.RdBu_r

    # one big grid query: for k = best fit + each overlay value, evaluate every draw at every logQ
    LQ = np.broadcast_to(lq, (nd, nq))
    def _pred(kvals):
        """kvals: array of k values; returns dict ratio_key -> (len(kvals), nq) posterior-median curves."""
        out = {}
        for key in keys:
            A = draws[:, Acol[key]][:, None] if nA else 1.0
            arr = np.empty((len(kvals), nq))
            for m, kv in enumerate(kvals):
                kk = np.full((nd, nq), kv) if kv is not None else np.broadcast_to(draws[:, 0][:, None], (nd, nq))
                qy = np.stack([LQ, kk,
                               np.broadcast_to(draws[:, 1][:, None], (nd, nq)),
                               np.broadcast_to(draws[:, 2][:, None], (nd, nq)),
                               np.broadcast_to(draws[:, 3][:, None], (nd, nq))], axis=-1).reshape(-1, 5)
                g = grid_interp(qy)[:, scol[key]].reshape(nd, nq)
                arr[m] = np.percentile(A * g, 50, axis=0)
            out[key] = arr
        return out
    band = {}
    for key in keys:
        A = draws[:, Acol[key]][:, None] if nA else 1.0
        qy = np.stack([LQ, np.broadcast_to(draws[:, 0][:, None], (nd, nq)),
                       np.broadcast_to(draws[:, 1][:, None], (nd, nq)),
                       np.broadcast_to(draws[:, 2][:, None], (nd, nq)),
                       np.broadcast_to(draws[:, 3][:, None], (nd, nq))], axis=-1).reshape(-1, 5)
        g = grid_interp(qy)[:, scol[key]].reshape(nd, nq)
        band[key] = np.percentile(A * g, [16, 50, 84], axis=0)
    over = _pred(list(k_over))

    fig, axes = plt.subplots(len(keys), 1, figsize=(9, 4.5 * len(keys)),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.subplots_adjust(left=0.18, right=0.85, top=0.88, bottom=0.08, hspace=0.05)
    for i, (ax, key) in enumerate(zip(axes, keys)):
        dt = tof[key]
        if dt not in data:
            continue
        lo, mid, hi = band[key]
        ax.fill_between(lq, lo, hi, color='0.6', alpha=0.35, lw=0, zorder=2,
                        label=r'$\rm 16-84\%~range$')
        for m, kv in enumerate(k_over):
            ax.plot(lq, over[key][m], '-', lw=3.0, color=k_cm(k_norm(kv)),
                    alpha=0.5, zorder=2.5)
        # "best fit" = posterior-predictive median curve (lies inside the 16-84 band by construction)
        # drawn in black: it is not a curve at a single k, so it takes no colour from the colour bar
        ax.plot(lq, mid, '-', lw=2.8, color='k', zorder=3, label=r'$\rm Posterior~median~fit$')
        d = data[dt]
        ax.errorbar(d['logQ'], d['R'], yerr=d['sig'], fmt='o', ms=5, color=col[key],
                    alpha=0.7, capsize=2, mec='k', mew=0.4, zorder=4)
        ax.set_yscale('log')
        ax.minorticks_on()
        lo_y, hi_y = ax.get_ylim()
        if np.log10(hi_y / lo_y) < 1.5:                      # under 1.5 dex: plain decimals, not 6x10^-1
            def _dec(v, _, minor):
                mant = v / 10 ** np.floor(np.log10(v) + 1e-9)
                return '' if minor and round(mant) not in (2, 3, 4, 6) else f'${v:g}$'
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, p: _dec(v, p, False)))
            ax.yaxis.set_minor_formatter(mpl.ticker.FuncFormatter(lambda v, p: _dec(v, p, True)))
        ax.tick_params(axis='both', which='major', top=True, right=True, length=8, width=1.5, direction='in', pad=5)
        ax.tick_params(axis='both', which='minor', top=True, right=True, length=4, width=1, direction='in')
        # line-ratio name inside the panel (replaces y-axis label)
        ax.text(0.03, 0.95, ylab[key], transform=ax.transAxes, va='top', ha='left',
                fontsize=15, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.6', alpha=0.85))
    axes[0].legend(loc='lower right', fontsize=13, frameon=False)
    axes[-1].set_xlabel(r'$\log Q~[\rm photons~s^{-1}]$')
    # y label just left of the widest tick label (tick-label width depends on the formatter)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    x_lab = min(ax.yaxis.get_tightbbox(rend).x0 for ax in axes) / fig.bbox.width
    fig.supylabel(r'$\rm Line~Ratio$', x=x_lab - 0.015, ha='right', fontsize=18)
    # top axis: observed log L_1350 = log Q + log C_Q + log 1350 (z-independent)
    _l1350_shift = LOG_C_Q_NAGAO + np.log10(1350.0)
    secax = axes[0].secondary_xaxis('top', functions=(lambda x: x + _l1350_shift,
                                                       lambda y: y - _l1350_shift))
    secax.set_xlabel(r'$\log L_{1350}~[\rm erg\,s^{-1}]$')
    secax.tick_params(axis='x', which='major', length=8, width=1.5, direction='in', pad=5)
    secax.tick_params(axis='x', which='minor', length=4, width=1, direction='in')
    secax.minorticks_on()
    axes[0].tick_params(axis='x', which='both', top=False)   # top ticks come from the L_1350 axis only
    # shared colour bar (replaces the legend)
    y0, y1 = axes[-1].get_position().y0, axes[0].get_position().y1   # same height as the panels
    cax = fig.add_axes([0.88, y0, 0.022, y1 - y0])
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=k_cm, norm=k_norm)
    cbar.set_label(r'${\rm \alpha-Metallicity~Gradient}, k$', rotation=270, labelpad=22)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)


def fit_object(rm_id='rm035', n_walkers=24, n_steps=2000, burn_in=800,
               out_dir='fits/alpha/loc_gradient', plot_dir='plots/alpha/loc_gradient',
               show_progress=True, grid_path=LOC_GRID_FILE, use_free_a=True,
               ratio_types=None, de_moves=False):
    """Fit the LOC + partial-breathing model to one object and save samples, best-fit
    parameters, a corner plot, and a model-vs-data plot.  Uses the precomputed model
    grid for speed (builds it on first use).  use_free_a=False drops the per-line
    normalizations A_MgII, A_SiIV (then the model predicts the absolute ratio level)."""
    import emcee
    from scipy.optimize import minimize
    corner_dir = os.path.join(plot_dir, 'corner')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(corner_dir, exist_ok=True)
    data, z = load_object_data(rm_id)
    if ratio_types:
        data = {t: d for t, d in data.items() if t in ratio_types}
    needed = set(ratio_types) if ratio_types else {'mg2_c4', 'si4_c4'}
    if not needed <= set(data):
        raise ValueError(f"{rm_id}: fit needs {sorted(needed)}; have {sorted(data)}")
    grid_interp = load_model_grid_interp(grid_path)        # used for the MCMC and the final plot
    names = param_names(use_free_a)
    ndim = len(names)
    n_walkers = max(int(n_walkers), 2 * ndim + 2)          # emcee needs > 2*ndim walkers
    print(f"{rm_id}: z={z:.4f} ({'free-A' if use_free_a else 'no-free-A'}); epochs: "
          + ", ".join(f"{t}={len(d['R'])}" for t, d in data.items()))

    # initial guess: k=0, p=0.35, log r_ref=17.5, Z_norm=3; A's from median(obs)/median(model)
    a_part = []
    if use_free_a:
        for t, col in (('mg2_c4', 0), ('si4_c4', 1)):
            d = data[t]
            g0 = float(grid_interp(np.array([[np.median(d['logQ']), 0.0, 0.35, 17.5, np.log10(3.0)]]))[0, col])
            a_part.append(float(np.clip(np.median(d['R']) / g0, *PARAM_BOUNDS['A_MgII'])))
    theta0 = np.array([0.0, 0.35, 17.5, np.log10(3.0)] + a_part + [np.log(0.15)])
    res = minimize(lambda th: -log_posterior(th, data, grid_interp), theta0,
                   method='Nelder-Mead', options={'maxiter': 3000, 'xatol': 1e-3, 'fatol': 1e-3})
    theta_opt = res.x if (res.success and np.isfinite(log_posterior(res.x, data, grid_interp))) else theta0
    # keep the init strictly inside the bounds
    for i, name in enumerate(names):
        lo, hi = PARAM_BOUNDS[name]
        theta_opt[i] = np.clip(theta_opt[i], lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo))
    print("  init: " + ", ".join(f"{n}={v:.3f}" for n, v in zip(names, theta_opt)))

    scale = np.maximum(np.abs(theta_opt) * 1e-2, 1e-3)
    pos = theta_opt + scale * np.random.randn(n_walkers, ndim)
    for i, name in enumerate(names):                       # keep all starting walkers inside the prior
        lo, hi = PARAM_BOUNDS[name]
        pos[:, i] = np.clip(pos[:, i], lo + 1e-4 * (hi - lo), hi - 1e-4 * (hi - lo))
    if show_progress:
        print(f"  running emcee: {n_walkers} walkers x {n_steps} steps (burn-in {burn_in})...")
    # differential-evolution moves propose jumps along walker-difference
    # vectors, letting walkers hop between separated modes; the default
    # stretch move cannot rescue a walker isolated in its own mode
    moves = ([(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)]
             if de_moves else None)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior_vec, args=(data, grid_interp),
                                    vectorize=True, moves=moves)
    sampler.run_mcmc(pos, n_steps, progress=show_progress)
    samples = sampler.get_chain(discard=burn_in, flat=True)

    med, std = np.median(samples, axis=0), np.std(samples, axis=0)
    print(f"\n{rm_id} best-fit (posterior median +/- std):")
    for n, m, s in zip(names, med, std):
        print(f"  {n:14s} = {m:+.4f} +/- {s:.4f}")
    print(f"  -> Z_norm = {10**med[3]:.3f} Z_sun ;  intrinsic scatter f = {np.exp(med[-1]):.3f}")

    np.savetxt(f"{out_dir}/{rm_id}_loc_mcmc_samples.txt", samples, header=" ".join(names))
    with open(f"{out_dir}/{rm_id}_loc_bestfit.txt", 'w') as fh:
        fh.write(f"# LOC + partial-breathing fit for {rm_id} (z = {z:.4f}); "
                 f"{'free-A' if use_free_a else 'no-free-A (A_MgII = A_SiIV = 1)'}\n")
        fh.write(f"# C_Q fixed = {C_Q_NAGAO:.4e} (log {LOG_C_Q_NAGAO:.4f}); Q_ref = {Q_REF:.1e}; "
                 f"Gamma = {GAMMA_LOC}, beta_LOC = {BETA_N_LOC}\n")
        for n, m, s in zip(names, med, std):
            fh.write(f"{n} = {m:.6f} +/- {s:.6f}\n")
        fh.write(f"Z_norm = {10**med[3]:.6f}\nf_int = {np.exp(med[-1]):.6f}\n")

    import corner
    fig = corner.corner(samples, labels=param_latex(use_free_a), truths=med, show_titles=True,
                        bins=24, smooth=1.0, smooth1d=1.0,
                        title_kwargs={'fontsize': 16}, label_kwargs={'fontsize': 20})
    for ax in fig.get_axes():
        ax.tick_params(labelsize=13)
    fig.savefig(f"{corner_dir}/{rm_id}_loc_corner.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    _plot_fit_result(rm_id, data, grid_interp, samples, f"{plot_dir}/{rm_id}_loc_fit.png")
    print(f"\nSaved: {out_dir}/{rm_id}_loc_{{mcmc_samples,bestfit}}.txt ; "
          f"{plot_dir}/{rm_id}_loc_fit.png ; {corner_dir}/{rm_id}_loc_corner.png")
    p16, p50, p84 = np.percentile(samples, [16, 50, 84], axis=0)
    return dict(rm_id=rm_id, z=z, n_epochs={t: len(d['R']) for t, d in data.items()},
                names=names, p16=p16, p50=p50, p84=p84, samples=samples)


def replot_object(rm_id, out_dir='fits/alpha/loc_gradient', plot_dir='plots/alpha/loc_gradient'):
    """Regenerate the corner + model-vs-data plots for an already-fitted object from its
    saved flat chain (`{rm_id}_loc_mcmc_samples.txt`) -- no MCMC rerun."""
    import corner
    corner_dir = os.path.join(plot_dir, 'corner')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(corner_dir, exist_ok=True)
    samples = np.loadtxt(os.path.join(out_dir, f"{rm_id}_loc_mcmc_samples.txt"))
    med = np.median(samples, axis=0)
    data, _ = load_object_data(rm_id)
    # units go on the axis labels; the per-panel titles stay unit-free so the
    # longer strings do not collide with neighbouring panels
    short = [lab.replace(r'~[\rm cm]', '').replace(r'~[Z_{\odot}]', '')
             for lab in param_latex(samples.shape[1] == 7)]
    fig = corner.corner(samples, labels=param_latex(samples.shape[1] == 7), truths=med, show_titles=True,
                        titles=short,
                        bins=24, smooth=1.0, smooth1d=1.0,
                        title_kwargs={'fontsize': 16}, label_kwargs={'fontsize': 20})
    for ax in fig.get_axes():
        ax.tick_params(labelsize=13)
    fig.savefig(f"{corner_dir}/{rm_id}_loc_corner.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    _plot_fit_result(rm_id, data, load_model_grid_interp(), samples, f"{plot_dir}/{rm_id}_loc_fit.png")
    print(f"replotted {rm_id} -> {plot_dir}/{rm_id}_loc_fit.png ; {corner_dir}/{rm_id}_loc_corner.png")


def replot_all(out_dir='fits/alpha/loc_gradient', plot_dir='plots/alpha/loc_gradient'):
    import glob
    ids = sorted(os.path.basename(p).split('_')[0]
                 for p in glob.glob(os.path.join(out_dir, 'rm*_loc_mcmc_samples.txt')))
    for i, rm_id in enumerate(ids, 1):
        print(f"[{i}/{len(ids)}]", end=' ')
        try:
            replot_object(rm_id, out_dir, plot_dir)
        except Exception as e:
            print(f"!! {rm_id} failed: {e}")


def _fit_all_worker(args):
    """Picklable worker for fit_all's process pool.  Returns (rm_id, row_or_None, err_or_None)."""
    rm_id, kw = args
    try:
        r = fit_object(rm_id, show_progress=False, **kw)
    except Exception as e:
        return rm_id, None, repr(e)
    row = {'rm_id': rm_id, 'z': r['z']}
    row.update({t: r['n_epochs'].get(t, 0) for t in ('mg2_c4', 'si4_c4')})
    for j, nm in enumerate(r['names']):
        row[nm] = r['p50'][j]
        row[f'{nm}_lo'] = r['p50'][j] - r['p16'][j]
        row[f'{nm}_hi'] = r['p84'][j] - r['p50'][j]
    row['Z_norm'] = 10 ** r['p50'][3]
    row['f_int'] = np.exp(r['p50'][-1])
    return rm_id, row, None


def fit_all(data_dir=_OBS_DATA_DIR, out_dir='fits/alpha/loc_gradient',
            plot_dir='plots/alpha/loc_gradient', n_walkers=28, n_steps=2500, burn_in=1000,
            skip_existing=True, n_procs=8, use_free_a=True, rm_ids=None):
    """Fit every rmNNN object found in `data_dir`, `n_procs` objects at a time.  Resumable:
    by default skips objects whose bestfit file already exists.  Writes a combined summary CSV."""
    import glob
    import multiprocessing as mp
    os.makedirs(out_dir, exist_ok=True)
    if rm_ids is None:
        rm_ids = sorted(os.path.basename(p).split('_')[0]
                        for p in glob.glob(os.path.join(data_dir, 'rm*_line_ratios.dat')))
    load_model_grid_interp()                                 # build the grid once up front (each worker re-loads it)
    todo = []
    for rm_id in rm_ids:
        if skip_existing and os.path.exists(os.path.join(out_dir, f"{rm_id}_loc_bestfit.txt")):
            continue
        d_chk, _ = load_object_data(rm_id)
        if not {'mg2_c4', 'si4_c4'} <= set(d_chk):
            print(f"{rm_id}: missing a ratio ({sorted(d_chk)}), skip joint fit")
            continue
        todo.append(rm_id)
    print(f"{len(todo)} object(s) to fit, {n_procs} at a time ({'free-A' if use_free_a else 'no-free-A'}).")
    summary_path = os.path.join(out_dir, 'loc_fit_summary.csv')
    kw = dict(n_walkers=n_walkers, n_steps=n_steps, burn_in=burn_in, out_dir=out_dir,
              plot_dir=plot_dir, use_free_a=use_free_a)
    rows, done = [], 0
    with mp.Pool(processes=max(1, int(n_procs))) as pool:
        for rm_id, row, err in pool.imap_unordered(_fit_all_worker, [(r, kw) for r in todo]):
            done += 1
            if err is not None:
                print(f"[{done}/{len(todo)}] {rm_id}: FAILED -- {err}")
                continue
            rows.append(row)
            pd.DataFrame(rows).to_csv(summary_path, index=False)    # checkpoint
            print(f"[{done}/{len(todo)}] {rm_id}: done  (k={row['k']:+.3f}, p={row['p']:.3f})")
    print(f"\nDone. {len(rows)} objects fitted this run. Summary -> {summary_path}")
    return summary_path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='LOC + partial-breathing model: forward plots, or MCMC fit.')
    ap.add_argument('--fit', metavar='RM_ID', nargs='?', const='rm035', default=None,
                    help='Run the MCMC fit for this object (default rm035 if no id given). '
                         'If omitted, only the forward-model demo plots are produced.')
    ap.add_argument('--nwalkers', type=int, default=28)
    ap.add_argument('--nsteps', type=int, default=2500)
    ap.add_argument('--burn', type=int, default=1000)
    ap.add_argument('--build-grid', action='store_true',
                    help='(Re)build the precomputed model grid used by the MCMC, then exit.')
    ap.add_argument('--fit-all', action='store_true',
                    help='Fit every rmNNN object in the data directory (resumable; writes a summary CSV).')
    ap.add_argument('--redo', action='store_true', help='With --fit-all: re-fit objects even if output exists.')
    ap.add_argument('--ncores', type=int, default=8, help='With --fit-all: number of objects to fit in parallel.')
    ap.add_argument('--replot', metavar='RM_ID', nargs='?', const='__all__', default=None,
                    help='Regenerate plots from saved chains (one RM_ID, or all fitted objects if no id). No MCMC rerun.')
    ap.add_argument('--free-a', action='store_true',
                    help='Include per-line normalizations A_MgII, A_SiIV as free parameters. '
                         'Reads/writes fits|plots/alpha/loc_gradient/.  Default: no free A (writes to .../loc_gradient_noA/).')
    ap.add_argument('--tag', default='',
                    help='Optional extra suffix on the output directories, e.g. --tag subsolar '
                         'writes to fits|plots/alpha/loc_gradient_noA_subsolar/.')
    ap.add_argument('--kmax', type=float, default=None,
                    help='Half-width of the k prior and model grid (default 1, the published fits). '
                         'Must be given both when building the grid and when fitting.')
    ap.add_argument('--fit-ids', default=None,
                    help='With --fit-all: restrict to a specific subset.  Either a comma-separated '
                         'list (rm142,rm616,...) or @path/to/file.txt (one rm_id per line).')
    args = ap.parse_args()
    if args.kmax is not None and float(os.environ.get('LOC_KMAX', '1.0')) != args.kmax:
        # module-level constants (prior, grid, grid file name) depend on KMAX:
        # restart with it in the environment so this process and its workers agree
        import sys
        os.environ['LOC_KMAX'] = str(args.kmax)
        entry = ['-m', __spec__.name] if __spec__ else [sys.argv[0]]
        os.execv(sys.executable, [sys.executable] + entry + sys.argv[1:])
    _fit_ids = None
    if args.fit_ids:
        if args.fit_ids.startswith('@'):
            with open(args.fit_ids[1:]) as fh:
                _fit_ids = [ln.strip() for ln in fh if ln.strip() and not ln.startswith('#')]
        else:
            _fit_ids = [s.strip() for s in args.fit_ids.split(',') if s.strip()]
    _suf  = '' if args.free_a else '_noA'
    _tag  = f'_{args.tag.lstrip("_")}' if args.tag else ''
    _od, _pd = f'fits/alpha/loc_gradient{_suf}{_tag}', f'plots/alpha/loc_gradient{_suf}{_tag}'
    if args.build_grid:
        build_model_grid()
    elif args.replot is not None:
        if args.replot == '__all__':
            replot_all(out_dir=_od, plot_dir=_pd)
        else:
            replot_object(args.replot, out_dir=_od, plot_dir=_pd)
    elif args.fit_all:
        fit_all(out_dir=_od, plot_dir=_pd, n_walkers=args.nwalkers, n_steps=args.nsteps,
                burn_in=args.burn, skip_existing=not args.redo, n_procs=args.ncores,
                use_free_a=args.free_a, rm_ids=_fit_ids)
    elif args.fit is not None:
        fit_object(args.fit, n_walkers=args.nwalkers, n_steps=args.nsteps, burn_in=args.burn,
                   out_dir=_od, plot_dir=_pd, use_free_a=args.free_a)
    else:
        plot_model_demo(z_norm=3.0, log_r_ref=17.5, out=f'{_pd}/model_ratio_vs_Q.png')
        plot_model_2d(z_norm=3.0, p_fixed=0.4, logrref_fixed=17.0, logrref_lo=17.0, logrref_hi=18.0,
                      p_fixed_c=0.4, out=f'{_pd}/model_2d_ratio_maps.png')
        plot_model_vs_data(z_norm=3.0, logrref_fixed=17.5,
                           logrref_values=(17.0, 17.5, 18.0, 18.5),
                           out=f'{_pd}/model_vs_data_ratio_vs_Q.png')
