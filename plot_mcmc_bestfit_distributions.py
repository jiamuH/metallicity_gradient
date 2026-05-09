"""
Plot best-fit parameter distributions and Z(r) profiles from MCMC fits.

Usage:
    python plot_mcmc_bestfit_distributions.py

Output:
    - mcmc_plots/bestfit_distributions.png
    - mcmc_plots/bestfit_Z_profiles.png
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.legend_handler

# Configuration
bestfit_dir = 'mcmc_fits'
plot_dir = 'mcmc_plots'
# r grid for Z(r) plotting — extended to small radii to overlap with nitrogen data
# (N5 at 2 ld ≈ 10^15.7 cm, N3 at ~90 ld ≈ 10^17.4 cm)
log_rin, log_rout = 14.5, 20.5
r = np.logspace(log_rin, log_rout, 400)
Z_min, Z_max = 1.0, 20.0
cm_to_pc = 3.086e18

plt.rcParams.update({
    'text.usetex': True,
    'axes.linewidth': 2,
    'font.family': 'serif',
    'font.weight': 'heavy',
    'font.size': 20,
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \boldmath'

os.makedirs(plot_dir, exist_ok=True)


def create_Z_profile(r, k, Z_0, r_ref):
    """Create metallicity profile Z(r) = Z_0 * (r / r_ref)^k.

    Z_0 is the metallicity at r_ref, k is the power-law gradient.
    This parameterization decouples the level (Z_0) from the slope (k).
    """
    return Z_0 * (r / r_ref)**k


def parse_bestfit_files(bestfit_dir):
    """Parse all bestfit files and return parameter arrays."""
    bestfit_files = sorted(glob.glob(os.path.join(bestfit_dir, 'rm*_bestfit.txt')))

    data = {'rm_id': [], 'k': [], 'k_err': [], 'beta': [], 'beta_err': [],
            'log_C_Q': [], 'log_C_Q_err': [], 'offset_mg2': [], 'offset_mg2_err': [],
            'offset_si4': [], 'offset_si4_err': [], 'rref': []}

    for f in bestfit_files:
        rm_id = os.path.basename(f).replace('_bestfit.txt', '')
        params = {}
        rref_val = 18.65  # default
        for line in open(f):
            if line.startswith('# rref'):
                rref_val = float(line.split('=')[1].split('(')[0].strip())
            if '=' in line and '+/-' in line and not line.startswith('#'):
                key = line.split('=')[0].strip()
                val = float(line.split('=')[1].split('+/-')[0].strip())
                err = float(line.split('+/-')[1].strip().split()[0])
                params[key] = (val, err)

        if 'k_joint' not in params:
            continue

        data['rm_id'].append(rm_id)
        data['rref'].append(rref_val)
        data['k'].append(params['k_joint'][0])
        data['k_err'].append(params['k_joint'][1])
        data['beta'].append(params['beta_joint'][0])
        data['beta_err'].append(params['beta_joint'][1])
        data['log_C_Q'].append(params['log_C_Q_joint'][0])
        data['log_C_Q_err'].append(params['log_C_Q_joint'][1])
        # offset_mg2 may not exist for si4_only fits
        if 'offset_mg2_joint' in params:
            data['offset_mg2'].append(params['offset_mg2_joint'][0])
            data['offset_mg2_err'].append(params['offset_mg2_joint'][1])
        else:
            data['offset_mg2'].append(np.nan)
            data['offset_mg2_err'].append(np.nan)
        data['offset_si4'].append(params['offset_si4_joint'][0])
        data['offset_si4_err'].append(params['offset_si4_joint'][1])

    for key in data:
        if key != 'rm_id':
            data[key] = np.array(data[key])

    return data


def plot_distributions(data, plot_dir):
    """Plot best-fit parameter distributions (|k| < 0.2 only)."""
    mask = np.abs(data['k']) < 0.2
    N = int(np.sum(mask))
    print(f"  Distributions: using {N}/{len(data['k'])} objects with |k| < 0.2")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    k_f = data['k'][mask]
    beta_f = data['beta'][mask]
    log_C_Q_f = data['log_C_Q'][mask]
    offset_mg2_f = data['offset_mg2'][mask]
    offset_si4_f = data['offset_si4'][mask]

    # k distribution
    ax = axes[0, 0]
    ax.hist(k_f, bins=30, color='steelblue', edgecolor='k', alpha=0.8)
    ax.set_xlabel(r'$k_\alpha\ \rm (metallicity\ gradient)$')
    ax.set_ylabel(r'$\rm Count$')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, which='both', direction='in')

    # beta distribution
    ax = axes[0, 1]
    ax.hist(beta_f, bins=30, color='coral', edgecolor='k', alpha=0.8)
    ax.set_xlabel(r'$\beta\ \rm (breathing\ factor)$')
    ax.set_ylabel(r'$\rm Count$')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, which='both', direction='in')

    # log_C_Q distribution
    ax = axes[0, 2]
    ax.hist(log_C_Q_f, bins=30, color='mediumpurple', edgecolor='k', alpha=0.8)
    ax.set_xlabel(r'$\log C_Q$')
    ax.set_ylabel(r'$\rm Count$')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, which='both', direction='in')

    # offset_mg2 distribution (skip if all NaN — si4_only mode)
    ax = axes[1, 0]
    mg2_valid = offset_mg2_f[~np.isnan(offset_mg2_f)]
    if len(mg2_valid) > 0:
        ax.hist(mg2_valid, bins=30, color='forestgreen', edgecolor='k', alpha=0.8)
        ax.set_xlabel(r'$A_{\rm MgII}$')
    else:
        ax.text(0.5, 0.5, r'$\rm N/A\ (si4\_only)$', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_xlabel(r'$A_{\rm MgII}$')
    ax.set_ylabel(r'$\rm Count$')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, which='both', direction='in')

    # offset_si4 distribution
    ax = axes[1, 1]
    ax.hist(offset_si4_f, bins=30, color='goldenrod', edgecolor='k', alpha=0.8)
    ax.set_xlabel(r'$A_{\rm SiIV}$')
    ax.set_ylabel(r'$\rm Count$')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, which='both', direction='in')

    # k vs beta scatter
    ax = axes[1, 2]
    sc = ax.scatter(k_f, beta_f, c=log_C_Q_f, cmap='viridis',
                    s=40, alpha=0.7, edgecolors='none')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r'$\log C_Q$')
    ax.set_xlabel(r'$k_\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, which='both', direction='in')
    plt.tight_layout()
    outfile = os.path.join(plot_dir, 'bestfit_distributions.png')
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {outfile}")


def load_nitrogen_data(filename='nitrogen_abundance_vs_r.dat'):
    """Load nitrogen abundance data from compute_nitrogen_abundance.py output."""
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Run compute_nitrogen_abundance.py first.")
        return None

    data = {}
    for line in open(filename):
        if line.startswith('#'):
            continue
        parts = line.split()
        species, r_ld, r_lo, r_hi, model, Zb, Zl, Zh = (
            parts[0], float(parts[1]), float(parts[2]), float(parts[3]),
            parts[4], float(parts[5]), float(parts[6]), float(parts[7]))
        if model not in data:
            data[model] = {}
        data[model][species] = {
            'r_ld': r_ld, 'r_lo_ld': r_lo, 'r_hi_ld': r_hi,
            'Z_best': Zb, 'Z_low': Zl, 'Z_high': Zh
        }
    return data


def plot_Z_profiles(data, plot_dir):
    """Plot Z(r) profiles (left axis, linear) with N/(C+O) nitrogen data (right axis, linear)."""
    N = len(data['rm_id'])
    k_values = data['k']
    rref_values = data['rref']
    ld_to_pc = 3e10 * 86400 / (3.086e18)  # 1 light-day in pc

    fig, ax = plt.subplots(figsize=(10, 7))
    from matplotlib.lines import Line2D

    # --- Load nitrogen data first to determine r range ---
    nitro = load_nitrogen_data()
    mkey = 'm1'
    if nitro is not None:
        all_r_lo_pc = [nitro[mkey][sp]['r_lo_ld'] * ld_to_pc for sp in ['N3', 'N4', 'N5']]
        all_r_hi_pc = [nitro[mkey][sp]['r_hi_ld'] * ld_to_pc for sp in ['N3', 'N4', 'N5']]
        # Model curves span data error bar range + small padding
        model_log_r_min = np.log10(min(all_r_lo_pc)) - 0.1
        model_log_r_max = np.log10(max(all_r_hi_pc)) + 0.1
    else:
        model_log_r_min = np.log10(r[0] / cm_to_pc)
        model_log_r_max = np.log10(r[-1] / cm_to_pc)

    # r grid for model curves — clipped to data extent
    r_plot_pc = np.logspace(model_log_r_min, model_log_r_max, 400)
    r_plot_cm = r_plot_pc * cm_to_pc
    log_r_pc = np.log10(r_plot_pc)

    # --- Left axis: Z(r) = Z_0 * (r/r_ref)^k from MCMC ---
    # Anchor Z_0 so that Z ≈ 3 Z_sun at the geometric center of the N data
    Z_anchor = 3.0

    # Filter out k >= 0.2 (keep flat and negative-gradient objects)
    k_cut = 0.2
    mask = k_values < k_cut
    k_filtered = k_values[mask]
    rref_filtered = rref_values[mask]
    print(f"  Using {len(k_filtered)}/{len(k_values)} objects with k < {k_cut}")

    # Use percentiles of k to define band edges, with envelope for no-crossing
    k_median = np.median(k_filtered)
    rref_median = np.median(rref_filtered)
    r_ref_cm = 10**rref_median
    # Re-anchor: Z_0 at r_ref such that Z(r_mid) = Z_anchor
    r_mid_pc = 10**(0.5 * (model_log_r_min + model_log_r_max))
    r_mid_cm = r_mid_pc * cm_to_pc
    Z_0 = Z_anchor / (r_mid_cm / r_ref_cm)**k_median
    print(f"  Z_0 = {Z_0:.2f} at r_ref (anchored to Z={Z_anchor} at geometric center)")
    Z_median_arr = create_Z_profile(r_plot_cm, k_median, Z_0, r_ref_cm)
    k_16 = np.percentile(k_filtered, 16)
    k_84 = np.percentile(k_filtered, 84)
    Z_k16 = create_Z_profile(r_plot_cm, k_16, Z_0, r_ref_cm)
    Z_k84 = create_Z_profile(r_plot_cm, k_84, Z_0, r_ref_cm)
    Z_16_arr = np.minimum(Z_k16, Z_k84)
    Z_84_arr = np.maximum(Z_k16, Z_k84)

    ax.plot(log_r_pc, Z_median_arr, color='dodgerblue', linewidth=3)
    ax.fill_between(log_r_pc, Z_16_arr, Z_84_arr,
                     color='dodgerblue', alpha=0.25)

    ax.set_xlabel(r'$\log r\ \rm [pc]$')
    ax.set_ylabel(r'$[\mathrm{N{+}C{+}O}] / [\mathrm{N{+}C{+}O}]_\odot$', color='dodgerblue')
    ax.tick_params(axis='y', which='both', colors='dodgerblue')

    # --- Top x-axis: gravitational radius ---
    # M_BH = 7e7 M_sun for NGC 5548
    M_BH = 7e7  # M_sun
    G_cgs = 6.674e-8  # cm^3 g^-1 s^-2
    M_sun_cgs = 1.989e33  # g
    c_cgs = 3e10  # cm/s
    r_g_cm = G_cgs * M_BH * M_sun_cgs / c_cgs**2  # gravitational radius in cm
    r_g_pc = r_g_cm / cm_to_pc  # gravitational radius in pc

    def pc_to_rg(log_r_pc_val):
        return log_r_pc_val - np.log10(r_g_pc)

    def rg_to_pc(log_r_rg_val):
        return log_r_rg_val + np.log10(r_g_pc)

    ax_top = ax.secondary_xaxis('top', functions=(pc_to_rg, rg_to_pc))
    ax_top.set_xlabel(r'$\log r\ [r_g]$')
    ax_top.minorticks_on()
    ax_top.tick_params(which='major', length=8, width=2, direction='in')
    ax_top.tick_params(which='minor', length=4, width=1, direction='in')

    # --- Right axis: [N/(C+O)] / [N/(C+O)]_sun from nitrogen models ---
    if nitro is not None:
        ax2 = ax.twinx()

        # Plot only m1 (1 Z_sun) nitrogen data — all orangered, different markers
        nitro_markers = {'N3': 's', 'N4': 'D', 'N5': 'o'}
        nitro_labels = {'N3': r'$\rm NIII]$', 'N4': r'$\rm NIV]$', 'N5': r'$\rm NV$'}
        Zm = 1.0

        r_pts_log, nco_pts, nco_lo_pts, nco_hi_pts = [], [], [], []
        for species in ['N3', 'N4', 'N5']:
            d = nitro[mkey][species]
            r_pc_val = d['r_ld'] * ld_to_pc
            log_r_pc_val = np.log10(r_pc_val)

            nco = d['Z_best'] / Zm
            nco_lo = d['Z_low'] / Zm
            nco_hi = d['Z_high'] / Zm

            r_pts_log.append(log_r_pc_val)
            nco_pts.append(nco)
            nco_lo_pts.append(nco_lo)
            nco_hi_pts.append(nco_hi)

        r_pts_log = np.array(r_pts_log)
        nco_pts = np.array(nco_pts)
        nco_lo_pts = np.array(nco_lo_pts)
        nco_hi_pts = np.array(nco_hi_pts)

        # Symmetrize errors in log space, add 0.2 dex systematic in quadrature
        nco_err_log = 0.5 * (np.log10(np.maximum(nco_hi_pts, 1e-3))
                             - np.log10(np.maximum(nco_lo_pts, 1e-3)))
        nco_err_log = np.sqrt(nco_err_log**2 + 0.2**2)

        # Expanded error bars (symmetric in log space → asymmetric in linear)
        nco_lo_exp = 10**(np.log10(nco_pts) - nco_err_log)
        nco_hi_exp = 10**(np.log10(nco_pts) + nco_err_log)

        # Plot with expanded errors, capture handles for legend
        eb_handles = {}
        for i, species in enumerate(['N3', 'N4', 'N5']):
            d = nitro[mkey][species]
            r_lo_pc = d['r_lo_ld'] * ld_to_pc
            r_hi_pc = d['r_hi_ld'] * ld_to_pc
            xerr = [[r_pts_log[i] - np.log10(r_lo_pc)],
                    [np.log10(r_hi_pc) - r_pts_log[i]]]
            yerr = [[nco_pts[i] - nco_lo_exp[i]], [nco_hi_exp[i] - nco_pts[i]]]

            eb_handles[species] = ax2.errorbar(
                r_pts_log[i], nco_pts[i], xerr=xerr, yerr=yerr,
                fmt=nitro_markers[species], color='orangered',
                markersize=12, lw=2.5, capsize=5, capthick=2, zorder=10)

        # Best fit in log-log space
        coeffs = np.polyfit(r_pts_log, np.log10(nco_pts), 1)
        k_N = coeffs[0]
        b_N = coeffs[1]

        # MC sampling for fit uncertainty
        n_mc = 1000
        k_N_samples, b_N_samples = [], []
        for _ in range(n_mc):
            nco_sample = 10**(np.log10(nco_pts) + np.random.randn(len(nco_pts)) * nco_err_log)
            c = np.polyfit(r_pts_log, np.log10(nco_sample), 1)
            k_N_samples.append(c[0])
            b_N_samples.append(c[1])
        k_N_samples = np.array(k_N_samples)
        b_N_samples = np.array(b_N_samples)

        # Fit line clipped to nitrogen data error bar range
        r_fit_log = np.linspace(model_log_r_min, model_log_r_max, 100)

        # Best fit line — orangered solid for N(r)
        nco_fit = 10**(k_N * r_fit_log + b_N)
        ax2.plot(r_fit_log, nco_fit, color='orangered', ls='-', lw=3)

        # MC prediction band — includes intrinsic scatter so ~68% of data falls within
        nco_mc_log = np.array([k * r_fit_log + b for k, b in zip(k_N_samples, b_N_samples)])
        fit_median_log = np.median(nco_mc_log, axis=0)
        fit_std_log = np.std(nco_mc_log, axis=0)
        # Intrinsic scatter: use the observation errors (expanded) as prediction scatter
        sigma_pred = np.mean(nco_err_log)
        pred_std = np.sqrt(fit_std_log**2 + sigma_pred**2)
        nco_mc_16 = 10**(fit_median_log - pred_std)
        nco_mc_84 = 10**(fit_median_log + pred_std)
        ax2.fill_between(r_fit_log, nco_mc_16, nco_mc_84,
                         color='orangered', alpha=0.3)

        ax2.set_ylabel(r'$[\mathrm{N/(C{+}O)}] / [\mathrm{N/(C{+}O)}]_\odot$',
                       color='orangered')
        ax2.minorticks_on()
        ax2.tick_params(axis='y', which='major', length=8, width=2, direction='in',
                        colors='orangered')
        ax2.tick_params(axis='y', which='minor', length=4, width=1, direction='in',
                        colors='orangered')

        # --- Model legend (upper right) ---
        k_alpha_up = k_84 - k_median
        k_alpha_lo = k_median - k_16
        k_N_16 = np.percentile(k_N_samples, 16)
        k_N_84 = np.percentile(k_N_samples, 84)
        k_N_up = k_N_84 - k_N
        k_N_lo = k_N - k_N_16

        model_handles = [
            plt.Rectangle((0, 0), 1, 1, fc='dodgerblue', alpha=0.25),
            plt.Rectangle((0, 0), 1, 1, fc='orangered', alpha=0.3),
        ]
        model_labels = [
            rf'$k_\alpha \equiv d\ln Z / d\ln r = {k_median:.2f}^{{+{k_alpha_up:.2f}}}_{{-{k_alpha_lo:.2f}}}$',
            rf'$k_N \equiv d\ln N / d\ln r = {k_N:.2f}^{{+{k_N_up:.2f}}}_{{-{k_N_lo:.2f}}}$',
        ]
        leg_model = ax.legend(model_handles, model_labels, fontsize=16,
                              loc='upper right', framealpha=0.9)
        ax.add_artist(leg_model)

        # --- Data legend (lower left) — NV on top, smaller symbols ---
        from matplotlib.lines import Line2D
        data_handles = []
        for sp in ['N5', 'N4', 'N3']:
            data_handles.append(Line2D([], [], marker=nitro_markers[sp],
                                       color='orangered', ls='none', markersize=8))
        data_labels = [nitro_labels[sp] for sp in ['N5', 'N4', 'N3']]
        ax.legend(data_handles, data_labels, fontsize=16,
                  loc='lower left', framealpha=0.9)
    else:
        ax.legend(fontsize=16, loc='best', framealpha=0.9)

    ax.minorticks_on()
    ax.tick_params(top=False, right=False, axis='both', which='major',
                   length=8, width=2, direction='in')
    ax.tick_params(top=False, right=False, axis='both', which='minor',
                   length=4, width=1, direction='in')

    plt.tight_layout()
    outfile = os.path.join(plot_dir, 'bestfit_Z_profiles.png')
    plt.savefig(outfile, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {outfile}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot MCMC best-fit distributions')
    parser.add_argument('--bestfit-dir', type=str, default=bestfit_dir,
                       help=f'Directory with bestfit files (default: {bestfit_dir})')
    parser.add_argument('--plot-dir', type=str, default=plot_dir,
                       help=f'Output directory for plots (default: {plot_dir})')
    pargs = parser.parse_args()
    bestfit_dir = pargs.bestfit_dir
    plot_dir = pargs.plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    data = parse_bestfit_files(bestfit_dir)
    N = len(data['rm_id'])
    print(f"Loaded {N} objects")
    print(f"  k: median={np.median(data['k']):.3f}, 16-84th=[{np.percentile(data['k'],16):.3f}, {np.percentile(data['k'],84):.3f}]")
    print(f"  beta: median={np.median(data['beta']):.3f}")
    print(f"  log_C_Q: median={np.median(data['log_C_Q']):.2f}")
    mg2_valid = data['offset_mg2'][~np.isnan(data['offset_mg2'])]
    if len(mg2_valid) > 0:
        print(f"  A_MgII: median={np.median(mg2_valid):.3f}")
    else:
        print(f"  A_MgII: N/A (si4_only mode)")
    print(f"  A_SiIV: median={np.nanmedian(data['offset_si4']):.3f}")

    plot_distributions(data, plot_dir)
    plot_Z_profiles(data, plot_dir)
