"""3-panel (left-right) EW contour plot for N III] / N IV] / N V using a fixed-
metallicity Cloudy LOC grid at (log phi 17-23, log n_H 9-12, fixed N-scale).

Recipe (same as the old notebook):
    EW = line_intensity / nFnu(continuum at a nearby wavelength)
both read from the same absolute (4 pi J, erg / cm^2 / s) line-list file.

Bottom panel: dL/dlog phi ∝ <4πJ_line>_n × r^{γ+3} at fixed Q (γ=−1), each
curve normalized to its peak.  This is the LOC integrand in φ-space; it has a
real peak (the bare <4πJ>_n curve does not, because line emission monotonically
grows with φ).

With bottom='cumulative' the bottom panel instead shows the cumulative fraction of the
LOC luminosity over the full grid (Gamma = -1 bold, other Gamma faint), with the Gamma = -1 16/84%
points (dotted) and the luminosity-weighted mean over the full grid (red dashed) marked.

Run from repo root:
    python3 -m nitrogen.plot_loc_m3_N23_contours                    # default m=3
    python3 -m nitrogen.plot_loc_m3_N23_contours m1_N23             # other tags
    python3 -m nitrogen.plot_loc_m3_N23_contours m3_N23 cumulative  # cumulative bottom panel
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

if not hasattr(np, 'trapezoid'):
    np.trapezoid = np.trapz


mpl.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.weight': 'heavy',
    'font.size': 16, 'axes.linewidth': 2, 'lines.linewidth': 2,
    'axes.labelweight': 'heavy', 'text.latex.preamble': r'\usepackage{bm} \usepackage{amsmath} \boldmath',
})

# tag -> dict with the per-chunk file paths, each on a (phi_axis, n_axis) sub-grid
# all in /Users/jiamuh/c23.01/my_models/loc_metal_nitrogen/
TAG_INFO = {
    'm1_N23': dict(
        chunks=[
            ('strong_LOC_m1_N23_LineList_BLR_Fe2_flux.txt',       np.arange(17.0, 23.5, 0.5), np.arange(9.0, 12.5, 0.5)),
            ('strong_LOC_m1_N23_lown_LineList_BLR_Fe2_flux.txt',  np.arange(17.0, 23.5, 0.5), np.arange(7.0,  9.0, 0.5)),
            ('strong_LOC_m1_N23_lophi_LineList_BLR_Fe2_flux.txt', np.arange(16.0, 17.0, 0.5), np.arange(7.0, 12.5, 0.5)),
        ],
        title=r'$Z_{\alpha} = 1\,Z_{\odot}$,\, $N_{\rm H} = 10^{23}~{\rm cm^{-2}}$'),
    'm3_N23': dict(
        chunks=[
            ('strong_LOC_m3_N23_LineList_BLR_Fe2_flux.txt',       np.arange(17.0, 23.5, 0.5), np.arange(9.0, 12.5, 0.5)),
        ],
        title=r'$Z_{\alpha} = 3\,Z_{\odot}$,\, $N_{\rm H} = 10^{23}~{\rm cm^{-2}}$'),
}
GRID_DIR  = '/Users/jiamuh/c23.01/my_models/loc_metal_nitrogen'

LOG_C     = np.log10(3e10)

LINE_INFO = {
    'N3': dict(col='blnd 1750.00A', cont='nFnu 1793.44A',
               label=r'$\rm N\,III]\,1750$', cbar=r'$\rm \log\,EW\,(N\,III])$'),
    'N4': dict(col='blnd 1486.00A', cont='nFnu 1458.33A',
               label=r'$\rm N\,IV]\,1486$',  cbar=r'$\rm \log\,EW\,(N\,IV])$'),
    'N5': dict(col='blnd 1240.00A', cont='nFnu 1262.79A',
               label=r'$\rm N\,V\,1240$',    cbar=r'$\rm \log\,EW\,(N\,V)$'),
}


GAMMA_VALUES = [-1.0, -0.5, 0.0]         # LOC radial-cover exponents f(r) ∝ r^Γ to overlay in the bottom panel
BETA_FIXED   = -1.0                      # g(n) ∝ n^{β}; Baldwin 1995 / Korista convention


def _read_grid(path):
    df = pd.read_csv(path, sep='\t', header=0)
    return df[~df.iloc[:, 0].astype(str).str.contains('GRID_DELIMIT')].reset_index(drop=True)


def main(tag='m3_N23', bottom='gamma'):
    info = TAG_INFO[tag]
    out = f'plots/nitrogen/loc_grid_N3N4N5_{tag}' + ('_cumulative' if bottom == 'cumulative' else '') + '.png'
    print(f'tag = {tag}\n  output : {out}')

    # 1. read all chunks and fill a single sparse (phi_full, n_full) array per line.
    phi_set = sorted({float(x) for _, p, _ in info['chunks'] for x in p})
    n_set   = sorted({float(x) for _, _, n in info['chunks'] for x in n})
    PHI_AX  = np.array(phi_set)
    LOGN_AX = np.array(n_set)
    nP, nN  = len(PHI_AX), len(LOGN_AX)
    ew_all  = {ln: np.full((nP, nN), np.nan) for ln in LINE_INFO}
    print(f'  PHI_AX = {PHI_AX}\n  LOGN_AX = {LOGN_AX}')
    for fname, phi_ax, n_ax in info['chunks']:
        df = _read_grid(os.path.join(GRID_DIR, fname))
        assert len(df) == len(phi_ax) * len(n_ax), f'{fname}: {len(df)} rows != {len(phi_ax)*len(n_ax)}'
        for jp, phi in enumerate(phi_ax):
            ip_full = int(np.argmin(np.abs(PHI_AX - phi)))
            for in_, n in enumerate(n_ax):
                in_full = int(np.argmin(np.abs(LOGN_AX - n)))
                row = df.iloc[jp * len(n_ax) + in_]
                for ln, lc in LINE_INFO.items():
                    f_line = float(row[lc['col']])
                    f_cont = float(row[lc['cont']])
                    ew_all[ln][ip_full, in_full] = f_line / max(f_cont, 1e-30)
    # any cells still NaN are uncovered by the chunks; treat as 0 for plotting
    for ln in LINE_INFO:
        ew_all[ln] = np.nan_to_num(ew_all[ln], nan=0.0)

    # use mpl_toolkits to attach each colorbar as a sibling axes (keeps the contour width
    # fixed, so the lower panels align horizontally with the contour above them).
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import colormaps

    fig = plt.figure(figsize=(18, 7.5))
    gs  = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.05, wspace=0.08)
    axes_top  = [fig.add_subplot(gs[0, c]) for c in range(3)]
    axes_mid  = [fig.add_subplot(gs[1, c], sharex=axes_top[c]) for c in range(3)]
    gamma_cmap = colormaps['plasma'](np.linspace(0.15, 0.85, len(GAMMA_VALUES)))
    n_lin = 10 ** LOGN_AX

    # ---- pre-compute the LOC integrand f(r) and r_EW / r_lo / r_hi for each line ----
    # NOTE: standard LOC averages line FLUX, not EW (EW is a ratio with a per-cell-varying
    # continuum normalizer, so per-cell <EW>_n does not correspond to any LOC observable).
    # We use line intensity directly (rel-to-H-beta, as Cloudy writes it) -- proportional to
    # local line flux -- in the LOC integral.
    GAMMA_LOC = -1.0
    BETA_R    = -1.0
    R_NV_LD   = 2.0
    LD_PER_CM = 1.0 / 2.59020684e15
    # also build the line-flux grid (rel-to-H-beta) -- needed for the LOC radial integrand
    em_all = {ln: np.full_like(ew_all[ln], np.nan) for ln in LINE_INFO}
    for fname, phi_ax, n_ax in info['chunks']:
        df = _read_grid(os.path.join(GRID_DIR, fname))
        for jp, phi in enumerate(phi_ax):
            ip_full = int(np.argmin(np.abs(PHI_AX - phi)))
            for in_, n in enumerate(n_ax):
                in_full = int(np.argmin(np.abs(LOGN_AX - n)))
                row = df.iloc[jp * len(n_ax) + in_]
                for ln, lc in LINE_INFO.items():
                    em_all[ln][ip_full, in_full] = float(row[lc['col']])
    for ln in LINE_INFO:
        em_all[ln] = np.nan_to_num(em_all[ln], nan=0.0)

    n_wt_r = (10 ** LOGN_AX) ** (BETA_R + 1); n_wt_r /= n_wt_r.sum()
    logphi_dense = np.linspace(PHI_AX.min(), PHI_AX.max(), 600)
    log4pi       = np.log10(4 * np.pi)
    log_q_arb    = 54.0
    logr_dense   = (log_q_arb - log4pi - logphi_dense) / 2.0
    r_dense_cm   = 10 ** logr_dense
    rcalc = {}
    for ln in ('N3', 'N4', 'N5'):
        em_n_phi = np.sum(em_all[ln] * n_wt_r[None, :], axis=1)             # <line flux>_n(phi)
        em_d = np.interp(logphi_dense, PHI_AX, em_n_phi)
        f    = np.maximum(em_d, 0.0) * r_dense_cm ** (GAMMA_LOC + 2)        # LOC radial integrand
        order = np.argsort(r_dense_cm)
        r_s, f_s = r_dense_cm[order], f[order]
        L_tot = np.trapezoid(f_s, r_s)
        r_ew  = np.trapezoid(r_s * f_s, r_s) / L_tot
        cum   = np.concatenate([[0.0], np.cumsum(0.5 * (f_s[:-1] + f_s[1:]) * np.diff(r_s))])
        cum  /= cum[-1]
        r_lo  = np.interp(0.16, cum, r_s)
        r_hi  = np.interp(0.84, cum, r_s)
        rcalc[ln] = dict(r=r_s, f=f_s, r_ew=r_ew, r_lo=r_lo, r_hi=r_hi)
    scale = (R_NV_LD / LD_PER_CM) / rcalc['N5']['r_ew']
    for ln in rcalc:
        rcalc[ln]['r_ld']    = rcalc[ln]['r']    * scale * LD_PER_CM
        rcalc[ln]['r_ew_ld'] = rcalc[ln]['r_ew'] * scale * LD_PER_CM
        rcalc[ln]['r_lo_ld'] = rcalc[ln]['r_lo'] * scale * LD_PER_CM
        rcalc[ln]['r_hi_ld'] = rcalc[ln]['r_hi'] * scale * LD_PER_CM
    for col, ln in enumerate(['N3', 'N4', 'N5']):
        ax  = axes_top[col]
        axb = axes_mid[col]
        ew = ew_all[ln]                                                # (n_phi, n_n)
        z = np.log10(np.maximum(ew.T, 1e-30))
        levels = np.arange(-10.0, 1.01, 1.0)                           # min = -10, max = +1, 1-dex bands
        cp = ax.contourf(PHI_AX, LOGN_AX, z, levels=levels, cmap='viridis')
        # log U overlay
        Pm, Nm = np.meshgrid(PHI_AX, LOGN_AX)
        cs = ax.contour(PHI_AX, LOGN_AX, Pm - Nm - LOG_C, colors='k', alpha=0.5, linestyles=':')
        ax.clabel(cs, inline=True, fontsize=14)
        ip, in_ = np.unravel_index(np.argmax(ew), ew.shape)
        # star: luminosity-weighted mean (log phi, log n) over the full grid (log phi 17-23,
        # log n 9-12), weight per cell = flux x n^(beta+1) x r^(Gamma+3)
        pp, nn = np.meshgrid(PHI_AX, LOGN_AX, indexing='ij')
        w = em_all[ln] * (10 ** nn) ** (BETA_R + 1) * 10 ** (-(pp / 2.0) * (GAMMA_LOC + 3))
        w = np.maximum(w, 0.0)
        phi_w, n_w = np.sum(w * pp) / w.sum(), np.sum(w * nn) / w.sum()
        print(f'  {ln}: weighted mean log phi = {phi_w:.2f}, log n = {n_w:.2f}')
        ax.plot(PHI_AX[ip], LOGN_AX[in_], marker='D', markersize=14, linestyle='None',   # EW maximum
                color='w', mec='k', mew=1.2, zorder=5, clip_on=False)
        ax.plot(phi_w, n_w, marker='*', markersize=28, linestyle='None',
                color='r', mec='k', mew=0.8, zorder=5)
        ax.set_xlim(PHI_AX.min(), PHI_AX.max())
        ax.set_ylim(LOGN_AX.min(), LOGN_AX.max())
        ax.tick_params(labelbottom=False)                              # x-tick labels live on the bottom (shared) panel
        ax.text(0.04, 0.96, LINE_INFO[ln]['label'], transform=ax.transAxes, va='top', ha='left',
                fontsize=18, color='w',
                bbox=dict(boxstyle='round,pad=0.3', fc='0.15', ec='none', alpha=0.7))
        # attach cbar slot on every panel (keeps contour widths equal); only show on rightmost
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.08)
        if col == 2:
            fig.colorbar(cp, cax=cax, label=r'$\log\,\rm EW$', ticks=np.arange(-10, 1, 2))
        else:
            cax.set_visible(False)
        # match the bottom panel's right edge by adding (and hiding) a same-sized appendage
        cax_b = make_axes_locatable(axb).append_axes('right', size='4%', pad=0.08)
        cax_b.set_visible(False)
        # LOC integrand in phi space: dL/dlog phi ∝ <4πJ_line>_n × r^{γ+3}
        # at fixed Q (r ∝ φ^{-1/2}); β=-1 (Baldwin 1995), varying γ ∈ GAMMA_VALUES.
        # Each curve normalized to its own peak.
        em_cell = em_all[ln]
        r_at_phi = 10 ** ((log_q_arb - log4pi - PHI_AX) / 2.0)            # cm, at fixed Q
        wt = n_lin ** (BETA_FIXED + 1); wt /= wt.sum()                   # density weight (per d log n)
        em_loc = np.sum(em_cell * wt[None, :], axis=1)
        if bottom == 'cumulative':
            # cumulative fraction of the LOC luminosity over the analysis window; the adopted
            # Gamma = -1 in bold black with its 16/50/84% points, the other Gamma values faint
            # normalized over the full grid (log phi 17-23)
            lp = np.linspace(PHI_AX.min(), PHI_AX.max(), 600)
            for ig, gamma in enumerate(GAMMA_VALUES):
                dL = np.interp(lp, PHI_AX, em_loc) * 10 ** (-(lp / 2.0) * (gamma + 3))
                cum = np.concatenate([[0.0], np.cumsum(0.5 * (dL[:-1] + dL[1:]) * np.diff(lp))])
                cum /= cum[-1]
                print(f'  {ln}: Gamma = {gamma:+.1f}: fraction at log phi > 21 = {1 - np.interp(21.0, lp, cum):.2f}, '
                      f'16/50/84% at log phi = ' + ' '.join(f'{np.interp(q, cum, lp):.2f}' for q in (0.16, 0.5, 0.84)))
                if gamma == GAMMA_LOC:
                    axb.plot(lp, cum, '-', lw=3.0, color='k', zorder=3, label=rf'$\Gamma = {gamma:+.1f}$')
                    for q in (0.16, 0.84):
                        axb.axvline(np.interp(q, cum, lp), color='0.4', ls=':', lw=2)
                else:
                    axb.plot(lp, cum, '-', lw=3.0, color=gamma_cmap[ig], alpha=0.55,
                             label=rf'$\Gamma = {gamma:+.1f}$')
            axb.axvline(phi_w, color='r', ls='--', lw=2)
            axb.set_ylim(0, 1.05)
            gamma_list = []
        else:
            gamma_list = GAMMA_VALUES
        for ig, gamma in enumerate(gamma_list):
            dLdlogphi = em_loc * r_at_phi ** (gamma + 3)
            dL_norm = dLdlogphi / np.max(dLdlogphi)
            axb.plot(PHI_AX, np.log10(np.maximum(dL_norm, 1e-30)), '-', lw=4.0,
                     color=gamma_cmap[ig], alpha=0.55, label=rf'$\Gamma = {gamma:+.1f}$')
        axb.set_xlim(PHI_AX.min(), PHI_AX.max())
        axb.set_xlabel(r'$\rm \log\,\phi~[photons~cm^{-2}~s^{-1}]$')
        if col == 0:
            axb.set_ylabel(r'$L_{\rm line}(<\phi)/L_{\rm line}$' if bottom == 'cumulative'
                           else r'$\log\,(dL/d\log\phi)_{\rm norm}$')
        else:
            axb.tick_params(labelleft=False)
        if bottom != 'cumulative':
            axb.relim(); axb.autoscale(axis='y')
        axb.minorticks_on()
        axb.tick_params(axis='both', which='major', top=True, right=True, length=8, width=1.5, direction='in', pad=5)
        axb.tick_params(axis='both', which='minor', top=True, right=True, length=4, width=1, direction='in')
        if col == 2:
            axb.legend(fontsize=10, frameon=False, loc='upper left' if bottom == 'cumulative' else 'best', ncol=1)
        # also match the contour width by attaching a hidden right-pad to the mid panel
        cax_b = make_axes_locatable(axb).append_axes('right', size='4%', pad=0.08); cax_b.set_visible(False)
        rc = rcalc[ln]
        print(f'  {ln}: peak EW = {ew.max():.3e} at phi = {PHI_AX[ip]:.1f}, n_H = {LOGN_AX[in_]:.1f};'
              f'  r_EW = {rc["r_ew_ld"]:.2f} ld  [{rc["r_lo_ld"]:.2f}, {rc["r_hi_ld"]:.2f}]')
    axes_top[0].set_ylabel(r'$\log\,n_{\rm H}~[\rm cm^{-3}]$')
    fig.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


if __name__ == '__main__':
    main(*sys.argv[1:])
