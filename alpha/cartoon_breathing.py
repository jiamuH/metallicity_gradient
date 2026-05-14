"""Cartoons: the (Q_gamma, r_ref) -> r_Q breathing geometry for three p values.

Produces a pair of 3x3 figures sharing the same layout (rows = Q_gamma, cols = p):

  - `cartoon_breathing.png`   : annulus shaded by log phi(r; Q_gamma).
  - `cartoon_breathing_Z.png` : same panels shaded by Z(r; Q_gamma, k) for a fixed gradient k.

Concrete numbers used throughout: Q_ref = 1e56 s^-1, r_ref = 1e17 cm.  Run with:
    python3 -m alpha.cartoon_breathing
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.weight': 'heavy',
    'font.size': 16, 'axes.linewidth': 2, 'lines.linewidth': 2,
    'axes.labelweight': 'heavy', 'text.latex.preamble': r'\usepackage{bm} \usepackage{amsmath} \boldmath',
})

OUT_PHI = 'plots/alpha/cartoon_breathing.png'
OUT_Z   = 'plots/alpha/cartoon_breathing_Z.png'

# concrete reference values
Q_REF = 1e56                                 # photons s^-1
R_REF = 1e17                                 # cm
LOG_R_REF = np.log10(R_REF)
LOG_Q_REF = np.log10(Q_REF)
LOG_4PI = np.log10(4.0 * np.pi)
LOGX_LO, LOGX_HI = -1.0, 1.0               # LOC fractional-radius window

PVALS = [0.0, 0.3, 0.5]
QFACS = [0.1, 1.0, 10.0]                     # Q_gamma / Q_ref (small Q first => top row)


def log_phi(logx, dlogQ, p):
    return LOG_Q_REF - LOG_4PI - 2 * LOG_R_REF - 2 * logx + (1 - 2 * p) * dlogQ


def log_Z(logx, dlogQ, p, k, log_z_norm):
    """log10 Z(x; Q_gamma) = log10 Z_norm + k*log10 x + (p*k)*dlog10 Q."""
    return log_z_norm + k * logx + (p * k) * dlogQ


def make_figure(mode, out, k=0.3, z_norm=3.0, z_contours=(3.0, 4.0, 5.0)):
    """mode = 'phi' or 'Z'.  For mode='phi' the panels are also overlaid with Z(r)
    contours (concentric circles, since Z depends only on physical radius)."""
    fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.5), sharex=True, sharey=True)
    if mode == 'phi':
        cmap, norm = plt.cm.viridis, mpl.colors.Normalize(vmin=17.0, vmax=23.0)
        cbar_label = r'$\log\phi~[\rm photons~cm^{-2}~s^{-1}]$'
    else:                                                    # 'Z'
        cmap = plt.cm.plasma
        # log Z range chosen to span the panels for k = 0.3 around Z_norm = 3 Z_sun
        norm = mpl.colors.Normalize(vmin=np.log10(z_norm) - 0.6, vmax=np.log10(z_norm) + 0.6)
        cbar_label = r'$\log\,(Z/Z_{\odot})$'

    AX_LIM = 19.0                                            # just fits the largest annulus (3.16*sqrt(10**1.5) ~ 17.8)
    NGRID = 400
    xv = np.linspace(-AX_LIM, AX_LIM, NGRID)
    yv = np.linspace(-AX_LIM, AX_LIM, NGRID)
    X, Y = np.meshgrid(xv, yv)
    R = np.sqrt(X ** 2 + Y ** 2)

    for i, qf in enumerate(QFACS):                           # top row = lowest Q
        dlogQ = np.log10(qf)
        for j, p in enumerate(PVALS):
            ax = axes[i, j]
            r_Q_units = qf ** p
            r_Q_cm = r_Q_units * R_REF
            x_frac = R / r_Q_units
            in_loc = (x_frac >= 10 ** LOGX_LO) & (x_frac <= 10 ** LOGX_HI)
            with np.errstate(divide='ignore'):
                logx = np.log10(np.where(in_loc, x_frac, 1.0))
            field = (log_phi(logx, dlogQ, p) if mode == 'phi'
                     else log_Z(logx, dlogQ, p, k, np.log10(z_norm)))
            field = np.where(in_loc, field, np.nan)
            ax.pcolormesh(X, Y, field, cmap=cmap, norm=norm, shading='auto', rasterized=True)
            # for the phi figure, overlay Z(r) contours.  Z = z_norm * (r/r_ref)^k depends
            # only on physical radius, so the contour positions are panel-independent; we
            # mask them to the LOC annulus to keep the figure uncluttered.
            if mode == 'phi':
                # Z(r) depends only on physical radius and is panel-invariant; show the
                # contours across the whole axis (not just inside the LOC) so the reader
                # sees how the BLR window samples a fixed metallicity field
                with np.errstate(divide='ignore'):
                    logZ_field = np.log10(z_norm) + k * np.log10(np.where(R > 0, R, 1e-30))
                lvls = sorted(np.log10(z_contours))
                cs = ax.contour(X, Y, logZ_field, levels=lvls, colors='0.25',
                                linewidths=0.9, linestyles='--', alpha=0.8, zorder=3)
                ax.clabel(cs, fmt=lambda v: rf'$Z={10 ** v:.0f}$', fontsize=7, inline=True)
            # LOC inner/outer boundary outlines
            th = np.linspace(0, 2 * np.pi, 400)
            for rb in (10 ** LOGX_LO * r_Q_units, 10 ** LOGX_HI * r_Q_units):
                ax.plot(rb * np.cos(th), rb * np.sin(th), '-', color='0.35', lw=1.0, zorder=2)
            # r_ref circle (anchor radius) -- panel-invariant
            ax.plot(1.0 * np.cos(th), 1.0 * np.sin(th), '-', color='crimson', lw=1.2, zorder=3.5)
            # r_Q circle (breathing radius for this Q, p)
            ax.plot(r_Q_units * np.cos(th), r_Q_units * np.sin(th), '-', color='royalblue',
                    lw=1.2, zorder=3.5)
            ax.plot(0, 0, marker='*', ms=8, color='gold', mec='k', mew=0.8, zorder=4)
            ax.set_aspect('equal')
            ax.set_xlim(-AX_LIM, AX_LIM); ax.set_ylim(-AX_LIM, AX_LIM)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(rf'$p = {p:.1f}$', fontsize=16, pad=6)
            if j == 0:
                qlbl = {0.1: r'$Q_\gamma = 10^{55}~{\rm s^{-1}}$',
                        1.0: r'$Q_\gamma = 10^{56}~{\rm s^{-1}}$',
                        10.0: r'$Q_\gamma = 10^{57}~{\rm s^{-1}}$'}[qf]
                ax.text(-0.06, 0.5, qlbl, transform=ax.transAxes, rotation=90,
                        va='center', ha='center', fontsize=14)
            # per-panel annotations: r_Q (top-left, no box) + field value at x=1 (bottom-right)
            ax.text(0.02, 0.98, rf'$r_Q = {r_Q_cm:.1e}$\,cm',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8)
            if mode == 'phi':
                txt = r'$\log\phi(x{=}1)\!=\!' + f'{log_phi(0.0, dlogQ, p):.2f}$'
            else:
                z_x1 = 10 ** log_Z(0.0, dlogQ, p, k, np.log10(z_norm))
                txt = rf'$Z(x{{=}}1)\!=\!{z_x1:.2f}\,Z_{{\odot}}$'
            ax.text(0.98, 0.02, txt, transform=ax.transAxes, va='bottom', ha='right', fontsize=8)

    fig.subplots_adjust(left=0.05, right=0.88, top=0.95, bottom=0.05, wspace=0.02, hspace=0.02)
    cax = fig.add_axes([0.90, 0.10, 0.02, 0.80])
    mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm).set_label(cbar_label, rotation=270, labelpad=16)

    if mode == 'phi':
        foot = (r'$Q_{\rm ref}=10^{56}~{\rm s^{-1}}$,\quad $r_{\rm ref}=10^{17}~{\rm cm}$ (red), '
                r'$r_Q$ (blue);'
                rf'\quad dashed contours: $Z/Z_{{\odot}}$ at $Z_{{\rm norm}}={z_norm:g}$, $k={k:+.2f}$')
    else:
        foot = (r'$Q_{\rm ref}=10^{56}~{\rm s^{-1}}$,\quad $r_{\rm ref}=10^{17}~{\rm cm}$ (red), '
                r'$r_Q$ (blue),'
                rf'\quad $Z_{{\rm norm}}={z_norm:g}~Z_{{\odot}}$,\quad $k={k:+.2f}$')
    fig.text(0.5, 0.005, foot, ha='center', fontsize=11)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def main():
    make_figure('phi', OUT_PHI, k=-0.2, z_norm=5.0)
    make_figure('Z', OUT_Z, k=-0.2, z_norm=5.0)


if __name__ == '__main__':
    main()
