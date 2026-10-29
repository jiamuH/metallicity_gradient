#!/usr/bin/env python3
"""Schematic edge-on view with the abundance
gradients.

Radius axis is r/r_g for M_SMBH = 1e8 Msun:
    r_g = GM/c^2 = 1.48e13 cm ;  100 pc = 3.09e20 cm = 2.1e7 r_g.

"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse, Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

mpl.rcParams.update({
    'text.usetex': True, 'font.family': 'serif', 'font.weight': 'heavy',
    'font.size': 15, 'axes.linewidth': 2,
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{bm} \boldmath',
})

OUT = Path('plots/alpha/cartoon_blr_nlr.png')

# These are just demonstrative numbers for the cartoon.
LRMIN, LRMAX = 0.0, 8.0
LR_DISK_OUT  = 3.0
LR_TOR_IN    = 5.0
LR_TOR_OUT   = 6.0
TOR_TOP      = 0.86
LOG_1PC      = np.log10(2.09e5)
LOG_100PC    = np.log10(2.09e7)

COL_DISK, COL_INNER = '#8893ad', '#f4b64c'
COL_BLR_BG, COL_NLR_BG = '#cbe0f0', '#d6ead8'
COL_TOR             = '#6b4a2f'
COL_IMM,  COL_META  = 'orangered', 'royalblue'
COL_N,    COL_CNO   = 'orangered', 'royalblue'
COL_BLR_CL, COL_NLR_CL = '#6da3c8', '#79b69b'

# Same N-ion palette as nitrogen/plot_ionization_structure.py:
# CMAP(0.1) ~ N III] purple, CMAP(0.5) ~ N IV] pink, CMAP(0.9) ~ N V orange.
_NION_COLORS = [
    (0.15, 0.08, 0.40), (0.294, 0.161, 0.569), (0.529, 0.173, 0.635),
    (0.753, 0.212, 0.616), (0.918, 0.310, 0.533), (0.980, 0.471, 0.463),
    (0.965, 0.663, 0.478), (0.929, 0.851, 0.639),
]
N_ION_CMAP = LinearSegmentedColormap.from_list('nion', _NION_COLORS, N=256)


def Hdisk(lr):
    return 0.085 + 0.025 * (np.asarray(lr) - 2.0)


def stars_disk(ax, rng, lr0, lr1, n, color, size, peak=None, sigma=0.4):
    """If peak is given, x positions are sampled from a Gaussian (peak,
    sigma) clipped to [lr0, lr1]; otherwise uniform in [lr0, lr1]."""
    if peak is not None:
        lr = np.clip(rng.normal(peak, sigma, n), lr0, lr1)
    else:
        lr = rng.uniform(lr0, lr1, n)
    y = rng.uniform(-0.72, 0.72, n) * Hdisk(lr)
    ax.scatter(lr, y, marker='*', s=size, c=color, edgecolors='#3a2600',
               linewidths=0.8, zorder=5)


def stars_free(ax, rng, lr0, lr1, n, color, size, ymax, halo=False,
               peak=None, sigma=0.4):
    """If peak is given, x positions are sampled from a Gaussian (peak,
    sigma) WITHOUT clipping (stars may extend beyond lr0/lr1)."""
    if peak is not None:
        lr = rng.normal(peak, sigma, n)
    else:
        lr = rng.uniform(lr0, lr1, n)
    y = rng.uniform(-ymax, ymax, n)
    if halo:
        for xx, yy in zip(lr, y):
            ax.add_patch(Ellipse((xx, yy), 0.30, 0.13, color=color,
                                 alpha=0.22, lw=0, zorder=4))
    ax.scatter(lr, y, marker='*', s=size, c=color, edgecolors='#3a2600',
               linewidths=0.8, zorder=5)


def clouds(ax, rng, lr0, lr1, n, ymin, ymax, w, h, color=None, alpha=0.55,
           cmap=None, cmap_t_range=(0.0, 1.0)):
    """If cmap is given, each cloud is coloured by its x position in
    [lr0, lr1] mapped linearly to cmap_t_range (so lr0 -> t_range[0])."""
    for _ in range(n):
        x = rng.uniform(lr0, lr1)
        s = rng.choice([-1, 1])
        y = s * rng.uniform(ymin, ymax)
        ww = w * rng.uniform(0.7, 1.4)
        hh = h * rng.uniform(0.7, 1.4)
        if cmap is not None:
            t0, t1 = cmap_t_range
            t = t0 + (t1 - t0) * (x - lr0) / (lr1 - lr0)
            c = cmap(float(np.clip(t, 0, 1)))
        else:
            c = color
        ax.add_patch(Ellipse((x, y), ww, hh, color=c, alpha=alpha,
                             lw=0, zorder=2.5))


def main():
    rng = np.random.default_rng(4)
    fig = plt.figure(figsize=(13.6, 7.6))
    # outer: BH side (narrow) | main cartoon side (wide), with a gap for the
    # truncation slashes.  Inside the main side, split further into the
    # inner-disk strip (x=2..3) and the BLR+torus+NLR strip (x=3..8), with
    # NO gap so the cartoon is continuous.  The upper-panel gradient axis
    # sits only above the BLR+torus+NLR strip, so its left edge aligns with
    # x=3 (BLR inner edge) in the cartoon below.
    gs_outer = GridSpec(1, 2, width_ratios=[1, 22], wspace=0.035)
    gs_left  = gs_outer[0].subgridspec(2, 1, height_ratios=[1.0, 1.95],
                                       hspace=0.05)
    gs_right = gs_outer[1].subgridspec(2, 2, height_ratios=[1.0, 1.95],
                                       width_ratios=[1, 5],
                                       hspace=0.05, wspace=0)
    axc_L = fig.add_subplot(gs_left[1])
    axg   = fig.add_subplot(gs_right[0, 1])
    axc   = fig.add_subplot(gs_right[1, :])
    BH_XLIM = (-0.6, 0.6)

    # ---------- top panel: abundance gradients ----------
    lr = np.linspace(2.0, LRMAX, 600)
    # NLR (r >= torus): metamorphic stars enrich N and C+N+O together;
    # N comes out slightly above C+N+O (more MS-yield N than PMS-yield CNO)
    N_FAC = 1.5                     # N/C+N+O ratio in NLR (constant)
    R_BLR_IN = 3.0                  # inner BLR edge ~ 10^3 r_g
    CNO_TOR = 4.0                   # alpha at torus = NLR peak
    CNO_BLR_IN = 5.0                # alpha at r_blr,in
    N_BLR_IN = 15.0                 # N at r_blr,in
    # NLR (r >= torus): metamorphic stars enrich N and C+N+O together, so
    # the N/(C+N+O) ratio stays constant at N_FAC across the NLR.
    cno_meta = CNO_TOR * 10 ** (-0.32 * (lr - LR_TOR_IN))
    # BLR (r < torus): alpha falls mildly outward from ~5 (r_blr,in) to ~3
    # (torus) with large error; N falls outward from ~15 (r_blr,in) to
    # N_FAC*CNO_TOR (torus, continuous with NLR).
    cno_blr_slope = (np.log10(CNO_TOR) - np.log10(CNO_BLR_IN)) \
                    / (LR_TOR_IN - R_BLR_IN)
    cno_blr = CNO_TOR * 10 ** (cno_blr_slope * (lr - LR_TOR_IN))
    nit_blr_slope = (np.log10(N_FAC * CNO_TOR) - np.log10(N_BLR_IN)) \
                    / (LR_TOR_IN - R_BLR_IN)
    nit_blr = (N_FAC * CNO_TOR) * 10 ** (nit_blr_slope * (lr - LR_TOR_IN))
    cno = np.where(lr >= LR_TOR_IN, cno_meta, cno_blr)
    nit = np.where(lr >= LR_TOR_IN, N_FAC * cno_meta, nit_blr)
    cno_err = 0.30
    cno_lo = cno_blr * 10 ** (-cno_err)
    cno_hi = cno_blr * 10 ** (+cno_err)

    axg.axvline(R_BLR_IN, color='0.55', lw=1.0, ls=':', alpha=0.6, zorder=1)
    axg.axvline(LR_TOR_IN, color='0.55', lw=1.2, ls='--', alpha=0.7,
                zorder=1)
    # large uncertainty band on the BLR C+N+O gradient
    axg.fill_between(lr, cno_lo, cno_hi,
                     where=((lr >= R_BLR_IN) & (lr < LR_TOR_IN)),
                     color=COL_CNO, alpha=0.20, lw=0, zorder=1.5)
    # brush-pen / painting-style lines: thick + rounded + white outline
    fx_cno = [pe.Stroke(linewidth=9.5, foreground='white'), pe.Normal()]
    fx_n   = [pe.Stroke(linewidth=7.5, foreground='white'), pe.Normal()]
    # solid where measurable (BLR, NLR); dashed across the dusty torus
    m_blr = (lr >= R_BLR_IN)  & (lr <= LR_TOR_IN)
    m_tor = (lr >= LR_TOR_IN) & (lr <= LR_TOR_OUT)
    m_nlr = (lr >= LR_TOR_OUT)
    axg.plot(lr[m_blr], cno[m_blr], color=COL_CNO, lw=6.0,
             solid_capstyle='round', path_effects=fx_cno,
             label=r'$\rm C+N+O$', zorder=2)
    axg.plot(lr[m_nlr], cno[m_nlr], color=COL_CNO, lw=6.0,
             solid_capstyle='round', path_effects=fx_cno, zorder=2)
    axg.plot(lr[m_tor], cno[m_tor], color=COL_CNO, lw=4.5, ls='--',
             dashes=(3.5, 2.5), dash_capstyle='round',
             path_effects=fx_cno, zorder=2)
    axg.plot(lr[m_blr], nit[m_blr], color=COL_N, lw=4.5,
             solid_capstyle='round', path_effects=fx_n,
             label=r'$\rm N$', zorder=3)
    axg.plot(lr[m_nlr], nit[m_nlr], color=COL_N, lw=4.5,
             solid_capstyle='round', path_effects=fx_n, zorder=3)
    axg.plot(lr[m_tor], nit[m_tor], color=COL_N, lw=3.5, ls='--',
             dashes=(3.5, 2.5), dash_capstyle='round',
             path_effects=fx_n, zorder=3)
    axg.set_ylim(0, 18)
    axg.set_xlim(R_BLR_IN, LRMAX)
    axg.set_ylabel(r'$\rm Abundance~Rel.~Solar$', fontsize=14)
    axg.legend(fontsize=13, loc='upper right', frameon=False,
               handlelength=1.6)
    axg.minorticks_on()
    axg.tick_params(which='both', direction='in', top=True, right=True,
                    labelbottom=False)
    axg.text(6.5, 6.0,
             r'$\rm enrichment~by~metamorphic~stars$' '\n'
             r'$\rm N~\&~C+N+O~rise~together$',
             fontsize=9.5, ha='center', va='center', color='0.25')
    axg.text(3.65, 16.5,
             r'$\rm immortal~stars$' '\n'
             r'$\rm enrich~N$',
             fontsize=9.5, ha='center', va='center', color='0.25')

    # ---------- bottom panel: edge-on cartoon ----------
    axc.set_xlim(2.0, LRMAX)
    axc.set_ylim(-1.12, 1.02)
    axc_L.set_xlim(BH_XLIM)
    axc_L.set_ylim(-1.12, 1.02)

    # zone background
    axc.axvspan(2.0, LR_DISK_OUT, color='0.92', lw=0, zorder=0)
    axc.axvspan(LR_DISK_OUT, LR_TOR_IN, color=COL_BLR_BG, alpha=0.7,
                lw=0, zorder=0)
    axc.axvspan(6.0, LRMAX, color=COL_NLR_BG, alpha=0.7,
                lw=0, zorder=0)

    # accretion disk -- single warm colour, truncated at the torus
    lr_d = np.linspace(2.0, LR_TOR_IN, 200)
    H = Hdisk(lr_d)
    axc.fill_between(lr_d, -H, H, color=COL_INNER, lw=0, zorder=2)
    axc.plot(lr_d,  H, color='#7a5a00', lw=1.5, zorder=3)
    axc.plot(lr_d, -H, color='#7a5a00', lw=1.5, zorder=3)

    # SMBH on the BH-side panel
    axc_L.scatter([0], [0], s=160, c=COL_INNER, alpha=0.55, zorder=4)
    axc_L.scatter([0], [0], s=460, marker='o', c='k', zorder=6)
    axc_L.text(0.0, 0.20, r'$\rm SMBH$', fontsize=11.5, ha='center',
               va='bottom')
    axc_L.set_yticks([]); axc_L.set_xticks([])
    for sp in ('top', 'left', 'right'):
        axc_L.spines[sp].set_visible(False)

    # dusty torus
    H_in = Hdisk(LR_TOR_IN)
    n_pts = 40
    xs = np.linspace(LR_TOR_IN, LR_TOR_OUT, n_pts)
    t = (xs - LR_TOR_IN) / (LR_TOR_OUT - LR_TOR_IN)
    y_top = H_in + (TOR_TOP - H_in) * t ** 1.5
    upper_verts = list(zip(xs, y_top)) + [(LR_TOR_OUT, H_in)]
    lower_verts = list(zip(xs, -y_top)) + [(LR_TOR_OUT, -H_in)]
    for verts in (upper_verts, lower_verts):
        axc.add_patch(Polygon(verts, closed=True, fc=COL_TOR, ec='k',
                              lw=1.3, zorder=3.5))
        axc.add_patch(Polygon(verts, closed=True, fc='none', ec='k',
                              lw=0.8, hatch='....', zorder=3.6))

    # BLR clouds -- coloured by ion
    clouds(axc, rng, LR_DISK_OUT + 0.10, LR_TOR_IN - 0.10, 32,
           ymin=0.27, ymax=0.58, w=0.16, h=0.07,
           alpha=0.55, cmap=N_ION_CMAP, cmap_t_range=(0.9, 0.1))
    # N-ion sequence in the BLR: N^{4+} (innermost) -> N^{2+} (outermost)
    for x_pos, lbl in [(3.4, r'$\rm N^{4+}$'),
                       (4.1, r'$\rm N^{3+}$'),
                       (4.8, r'$\rm N^{2+}$')]:
        axc.text(x_pos, 0.20, lbl, ha='center', va='bottom',
                 fontsize=12, color='0.15', zorder=6)
    # NLR clouds
    clouds(axc, rng, LR_TOR_OUT + 0.05, 6.78, 4,
           ymin=0.42, ymax=0.82, w=0.22, h=0.10, color=COL_NLR_CL,
           alpha=0.40)
    clouds(axc, rng, LR_TOR_OUT + 0.05, 6.78, 2,
           ymin=0.05, ymax=0.28, w=0.21, h=0.09, color=COL_NLR_CL,
           alpha=0.30)
    clouds(axc, rng, 6.78, 7.0, 4,
           ymin=0.42, ymax=0.82, w=0.22, h=0.10, color=COL_NLR_CL,
           alpha=0.55)
    clouds(axc, rng, 7.0, LRMAX - 0.05, 20,
           ymin=0.42, ymax=0.82, w=0.22, h=0.10, color=COL_NLR_CL,
           alpha=0.55)
    clouds(axc, rng, 6.78, LRMAX - 0.05, 6,
           ymin=0.05, ymax=0.28, w=0.21, h=0.09, color=COL_NLR_CL,
           alpha=0.40)

    # embedded stars -- immortal stars: Gaussian peak at 10^3.5 r_g just for demonstration.
    stars_disk(axc, rng, LR_DISK_OUT + 0.05, LR_TOR_IN - 0.05, 17,
               COL_IMM, 200, peak=3.5, sigma=0.40)
    # metamorphic stars: Gaussian peak at 10^5.5 r_g (torus midplane);
    # tails extend outside the torus (some stars in BLR/NLR adjacent zones)
    stars_free(axc, rng, LR_TOR_IN + 0.05, LR_TOR_OUT - 0.05, 12,
               COL_META, 240, ymax=0.07, halo=True,
               peak=5.5, sigma=0.40)

    # zone labels
    lab = dict(ha='center', va='center', fontsize=16)
    axc.text(2.5, 0.93,
             r'$\rm inner$' '\n' r'$\rm disk$',
             fontsize=12, ha='center', va='center')
    axc.text((LR_DISK_OUT + LR_TOR_IN) / 2, 0.93, r'$\rm BLR$', **lab)
    axc.text((6.0 + LRMAX) / 2, 0.93, r'$\rm NLR$', **lab)
    axc.annotate(r'$\rm dusty~torus$', xy=(LR_TOR_OUT - 0.05, 0.78),
                 xytext=(LR_TOR_IN + 0.25, 1.00), ha='center', va='bottom',
                 fontsize=13, color=COL_TOR,
                 arrowprops=dict(arrowstyle='-', lw=1.3, color=COL_TOR))
    axc.annotate(r'$\rm accretion~disk$', xy=(2.85, -0.118),
                 xytext=(3.05, -0.47), fontsize=12.5, style='italic',
                 color='#3c465c', ha='center', va='center',
                 arrowprops=dict(arrowstyle='-', lw=1.2, color='#3c465c'))

    # callouts for the two stellar populations
    b1 = dict(boxstyle='round,pad=0.35', fc='white', ec='#7a5a00', lw=1.5)
    axc.annotate(r'\textbf{immortal stars}' '\n'
                 r'$\rm sustained~main~sequence;$' '\n'
                 r'$\rm winds~enrich~N$',
                 xy=(4.0, -0.18), xytext=(4.0, -0.85), ha='center',
                 va='center', fontsize=11.5, bbox=b1, zorder=10,
                 arrowprops=dict(arrowstyle='-|>', lw=1.8,
                                 color='#7a5a00'))
    b2 = dict(boxstyle='round,pad=0.35', fc='white', ec='#5a1c08', lw=1.5)
    axc.annotate(r'\textbf{metamorphic stars}' '\n'
                 r'$\rm generate~C+N+O+Fe$',
                 xy=((LR_TOR_IN + LR_TOR_OUT) / 2, 0.0),
                 xytext=(6.5, -0.88), ha='center',
                 va='center', fontsize=10, bbox=b2, zorder=10,
                 arrowprops=dict(arrowstyle='-|>', lw=1.8,
                                 color='#5a1c08'))

    # 1 pc and 100 pc markers (tick inside the panel, label above the axis)
    for lx, lab in [(LOG_1PC, r'$1~\rm pc$'), (LOG_100PC, r'$100~\rm pc$')]:
        axc.plot([lx, lx], [-0.80, -0.72], color='k', lw=1.8)
        axc.text(lx, -0.83, lab, ha='center', va='top', fontsize=11.5)

    axc.set_yticks([])
    for sp in ('top', 'left', 'right'):
        axc.spines[sp].set_visible(False)
    axc.set_xticks([2, 3, 4, 5, 6, 7, 8])
    axc.set_xticklabels([r'$10^{%d}$' % e for e in (2, 3, 4, 5, 6, 7, 8)])
    axc.tick_params(axis='x', direction='out', length=7, width=1.6)
    axc.set_xlabel(r'$\rm Distance~from~SMBH~$$[r_g]$', fontsize=14)

    # ---------- broken-axis slashes (bottom panel only) ----------
    d = 0.018
    sl = dict(color='k', lw=1.6, clip_on=False, zorder=20)
    axc_L.plot([1-d, 1+d], [-d, +d], transform=axc_L.transAxes, **sl)
    axc.plot(  [-d, +d],   [-d, +d], transform=axc.transAxes,   **sl)

    fig.savefig(OUT, dpi=300, bbox_inches='tight')
    print(f'Saved: {OUT}')
    plt.close(fig)


if __name__ == '__main__':
    main()
