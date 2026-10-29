"""Numerical check: for the no-A LOC model at fixed (k=0, p=0.35, Z_norm=3 Zsun),
find the log r_ref at which each model line ratio matches the Van den Berk (2001)
composite, and report whether MgII/CIV and (SiIV+OIV)/CIV can be satisfied at a
common r_ref.

Run from repo root:
    python3 -m alpha.check_vdb_rref_overlap
"""
import numpy as np
from alpha.loc_gradient_fit import (make_interpolators, model_line_ratios, OBS_RANGE,
                                    LOGQ_REF, _LOG4PI, LOGPHI_GRID)

K_FIX, P_FIX, Z_NORM = 0.0, 0.35, 3.0
LOGQ_AX = np.linspace(53.5, 56.0, 11)
LOGR_AX = np.arange(15.50, 19.001, 0.01)          # fine r_ref scan
LOGX_LO, LOGX_HI = -1.0, 1.0                      # LOC cloud distribution in log x (model default)


def report_clipping():
    """For a range of log r_ref, report logphi_ref and the fraction of the LOC
    cloud distribution that gets clipped against the Cloudy phi grid edges."""
    phi_min, phi_max = LOGPHI_GRID.min(), LOGPHI_GRID.max()
    logx = np.linspace(LOGX_LO, LOGX_HI, 400)
    print(f"\nphi-grid clipping  (LOGPHI_GRID = [{phi_min:.0f}, {phi_max:.0f}], "
          f"log x in [{LOGX_LO:.0f}, {LOGX_HI:.0f}] -> cloud phi spans logphi_ref +/- 2 dex)")
    print(f"  logphi_ref = {LOGQ_REF:.2f} - {_LOG4PI:.3f} - 2 log r_ref;  evaluated at Q = Q_ref")
    print(f"{'log r_ref':>10s} | {'logphi_ref':>11s} | {'clipped @hi':>11s} | {'clipped @lo':>11s}")
    print('-' * 52)
    for lr in (15.6, 16.0, 16.34, 16.7, 16.95, 17.5, 17.8, 17.95, 18.2):
        lpr = LOGQ_REF - _LOG4PI - 2.0 * lr
        logphi_cloud = lpr - 2.0 * logx
        f_hi = np.mean(logphi_cloud > phi_max)
        f_lo = np.mean(logphi_cloud < phi_min)
        print(f"{lr:10.2f} | {lpr:11.2f} | {100*f_hi:9.0f}\\% | {100*f_lo:9.0f}\\%")


def build_maps(interps, logQ_ax=LOGQ_AX):
    """Return {key: array(n_r, n_Q)} of log10(model ratio)."""
    maps = {key: np.zeros((LOGR_AX.size, logQ_ax.size)) for key in OBS_RANGE}
    for i, lr in enumerate(LOGR_AX):
        ratios, _, _ = model_line_ratios(interps, logQ_ax, K_FIX, P_FIX,
                                         log_r_ref=lr, z_norm=Z_NORM)
        for key in OBS_RANGE:
            maps[key][i] = np.log10(ratios[key])
    return maps


def crossings(model_col, target):
    """Interpolated log r_ref values where model_col crosses `target`."""
    d = model_col - target
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    out = []
    for i in idx:
        if model_col[i + 1] != model_col[i]:
            out.append(LOGR_AX[i] + 0.01 * (target - model_col[i]) /
                       (model_col[i + 1] - model_col[i]))
    return out


def match_intervals(model_col, obs, oerr):
    """List of (r_cen, r_lo, r_hi): each crossing of `obs` with the r_ref interval
    over which the model stays within obs +/- oerr around that crossing."""
    cen = crossings(model_col, obs)
    edges = sorted(crossings(model_col, obs - oerr) + crossings(model_col, obs + oerr))
    out = []
    for rc in cen:
        below = [e for e in edges if e <= rc]
        above = [e for e in edges if e >= rc]
        out.append((rc, max(below) if below else rc, min(above) if above else rc))
    return out


def _fmt(ivs):
    return ', '.join(f"{rc:.2f}[{lo:.2f}-{hi:.2f}]" for rc, lo, hi in ivs) if ivs else 'none'


def plot_rref_solutions(interps, out='plots/alpha/loc_gradient_noA/vdb_rref_solutions.png'):
    """Panel: the log r_ref at which each model ratio matches the VdB composite,
    vs log Q.  MgII/CIV is one branch; (SiIV+OIV)/CIV is double-valued, so it has
    a low and a high branch.  No intersection => no common r_ref."""
    import os
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({
        'text.usetex': True, 'font.family': 'serif', 'font.weight': 'heavy',
        'font.size': 16, 'axes.linewidth': 2, 'lines.linewidth': 2,
        'axes.labelweight': 'heavy',
        'text.latex.preamble': r'\usepackage{bm} \usepackage{amsmath} \boldmath'})

    logQ_fine = np.linspace(53.5, 56.0, 80)
    maps = build_maps(interps, logQ_fine)
    SPLIT = 16.8                                          # log r_ref dividing the two SiIV branches
    curves = {'mg': [], 'si_lo': [], 'si_hi': []}
    for j, lq in enumerate(logQ_fine):
        for rc, rl, rh in match_intervals(maps['MgII/CIV'][:, j], *OBS_RANGE['MgII/CIV']):
            curves['mg'].append((lq, rc, rl, rh))
        for rc, rl, rh in match_intervals(maps['(SiIV+OIV)/CIV'][:, j], *OBS_RANGE['(SiIV+OIV)/CIV']):
            curves['si_lo' if rc < SPLIT else 'si_hi'].append((lq, rc, rl, rh))

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    style = {'mg':    ('dodgerblue', r'$\rm Mg\,II / C\,IV = VdB$'),
             'si_lo': ('deeppink',   r'$\rm (Si\,IV + O\,IV) / C\,IV = VdB$'),
             'si_hi': ('deeppink',   None)}
    for name, (color, label) in style.items():
        if not curves[name]:
            continue
        arr = np.array(curves[name])
        q, rc, rl, rh = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        ax.fill_between(q, rl, rh, color=color, alpha=0.30, lw=0)
        ax.plot(q, rc, '-', color=color, label=label)
    ax.set_xlabel(r'$\log Q~[\rm photons~s^{-1}]$')
    ax.set_ylabel(r'$\log r_{\rm ref}~[\rm cm]$')
    ax.set_xlim(53.5, 56.0)
    ax.set_title(rf'$k={K_FIX:.2f},\ p={P_FIX:.2f},\ '
                 rf'Z_{{\rm norm}}={Z_NORM:g}\,Z_{{\odot}}$ (no $A$)', fontsize=15)
    ax.text(0.05, 0.55, r'$\rm no\ common\ \mathit{r}_{ref}$', transform=ax.transAxes,
            fontsize=15, color='0.2',
            bbox=dict(boxstyle='round,pad=0.3', fc='0.92', ec='0.5'))
    ax.legend(fontsize=12, frameon=False, loc='upper left')
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True,
                   length=8, width=1.5, direction='in', pad=5)
    ax.tick_params(axis='both', which='minor', top=True, right=True,
                   length=4, width=1, direction='in')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out}")


def main():
    interps = make_interpolators()
    maps = build_maps(interps)
    print(f"k={K_FIX}, p={P_FIX}, Z_norm={Z_NORM} Zsun;  r_ref scan "
          f"[{LOGR_AX[0]:.2f}, {LOGR_AX[-1]:.2f}]")
    for key, (obs, oerr) in OBS_RANGE.items():
        print(f"  VdB {key:16s}: log = {obs:+.3f} +/- {oerr:.3f}")
    print("\nEntries: r_cen[r_lo-r_hi]  (central crossing and +/-err band)\n")
    hdr = (f"{'logQ':>6s} | {'MgII/CIV  log r_ref':>24s} | "
           f"{'(SiIV+OIV)/CIV  log r_ref':>36s} | common r_ref?")
    print(hdr)
    print('-' * len(hdr))
    for j, lq in enumerate(LOGQ_AX):
        mg = match_intervals(maps['MgII/CIV'][:, j], *OBS_RANGE['MgII/CIV'])
        si = match_intervals(maps['(SiIV+OIV)/CIV'][:, j], *OBS_RANGE['(SiIV+OIV)/CIV'])
        overlap = any(a[1] <= b[2] and b[1] <= a[2] for a in mg for b in si)
        print(f"{lq:6.2f} | {_fmt(mg):>24s} | {_fmt(si):>36s} | "
              f"{'YES' if overlap else 'no'}")
    report_clipping()
    plot_rref_solutions(interps)


if __name__ == '__main__':
    main()
