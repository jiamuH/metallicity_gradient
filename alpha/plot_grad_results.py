#!/usr/bin/env python3
"""
Plot best-fit k values (metallicity gradient) as a function of mean F1350 flux.

This script:
1. Reads best-fit parameters from mcmc_fits/rm*_bestfit.txt files
2. Reads observed F1350 flux ranges from observed_line_ratio_data/rm*_line_ratios.dat files
3. Plots k vs mean F1350 with errorbars (k error on y-axis, F1350 min/max on x-axis)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import glob

# Set up matplotlib
plt.rcParams.update({
    'text.usetex': True,
    'axes.linewidth': 2,
    'font.family': 'serif',
    'font.weight': 'heavy',
    'font.size': 16,
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \boldmath'

# Configuration
bestfit_dir = 'fits/alpha/mcmc_fits'
observed_data_dir = 'data/alpha/observed_line_ratio_data'
output_dir = 'plots/alpha/grad_results_plots'
os.makedirs(output_dir, exist_ok=True)


def load_bestfit_params(filename):
    """
    Load best-fit parameters from a bestfit.txt file.
    
    Returns:
    --------
    params : dict
        Dictionary with 'k_joint', 'k_std_joint', etc., or None if not found
    """
    params = {}
    
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            # Parse lines like: "k_joint = 1.234567 ± 0.123456"
            # Also handle lines without spaces around = or ±
            if '=' in line:
                # Try to find the ± symbol (could be regular ± or unicode)
                if '±' in line or '\xb1' in line:
                    # Replace unicode ± with regular ±
                    line_clean = line.replace('\xb1', '±')
                    parts = line_clean.split('=')
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value_part = parts[1].strip()
                        # Split by ±
                        if '±' in value_part:
                            value_str, error_str = value_part.split('±')
                            try:
                                params[key] = float(value_str.strip())
                                params[key + '_std'] = float(error_str.strip())
                            except ValueError:
                                continue
    
    return params if len(params) > 0 else None


def load_f1350_range(filename):
    """
    Load F1350 flux range from observed data file.
    
    Returns:
    --------
    f1350_min, f1350_max, f1350_mean : float
        Minimum, maximum, and mean F1350 values, or None if not found
    """
    if not os.path.exists(filename):
        return None, None, None
    
    f1350_values = []
    
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) >= 1:
                try:
                    f1350 = float(parts[0])
                    f1350_values.append(f1350)
                except ValueError:
                    continue
    
    if len(f1350_values) == 0:
        return None, None, None
    
    f1350_values = np.array(f1350_values)
    return f1350_values.min(), f1350_values.max(), f1350_values.mean()


def main():
    """Main plotting routine."""
    # Find all bestfit files
    pattern = os.path.join(bestfit_dir, 'rm*_bestfit.txt')
    bestfit_files = sorted(glob.glob(pattern))
    
    if len(bestfit_files) == 0:
        print(f"No bestfit files found in {bestfit_dir}")
        return
    
    print(f"Found {len(bestfit_files)} bestfit files")
    
    # Collect data
    rm_ids = []
    k_values = []
    k_errors = []
    f1350_means = []
    f1350_mins = []
    f1350_maxs = []
    
    for bestfit_file in bestfit_files:
        # Extract rm_id from filename
        basename = os.path.basename(bestfit_file)
        match = re.match(r'(rm\d+)_bestfit\.txt', basename)
        if not match:
            continue
        
        rm_id = match.group(1)
        
        # Load best-fit parameters
        params = load_bestfit_params(bestfit_file)
        if params is None:
            print(f"Warning: Could not load parameters from {bestfit_file}")
            continue
        
        # Get k value from joint fit (preferred) or individual fits
        if 'k_joint' in params:
            k = params['k_joint']
            k_err = params.get('k_std_joint', 0.0)
        elif 'k_mg2_c4' in params:
            k = params['k_mg2_c4']
            k_err = params.get('k_std_mg2_c4', 0.0)
        elif 'k_si4_c4' in params:
            k = params['k_si4_c4']
            k_err = params.get('k_std_si4_c4', 0.0)
        else:
            print(f"Warning: No k value found in {bestfit_file}")
            continue
        
        # Load F1350 range from observed data
        observed_file = os.path.join(observed_data_dir, f'{rm_id}_line_ratios.dat')
        f1350_min, f1350_max, f1350_mean = load_f1350_range(observed_file)
        
        if f1350_mean is None:
            print(f"Warning: Could not load F1350 data from {observed_file}")
            continue
        
        rm_ids.append(rm_id)
        k_values.append(k)
        k_errors.append(k_err)
        f1350_means.append(f1350_mean)
        f1350_mins.append(f1350_min)
        f1350_maxs.append(f1350_max)
    
    if len(rm_ids) == 0:
        print("No valid data found to plot")
        return
    
    # Convert to numpy arrays
    rm_ids = np.array(rm_ids)
    k_values = np.array(k_values)
    k_errors = np.array(k_errors)
    f1350_means = np.array(f1350_means)
    f1350_mins = np.array(f1350_mins)
    f1350_maxs = np.array(f1350_maxs)
    
    # Calculate x errorbars (asymmetric)
    # Ensure mean is between min and max (should always be true, but check anyway)
    xerr_lower = np.maximum(0, f1350_means - f1350_mins)
    xerr_upper = np.maximum(0, f1350_maxs - f1350_means)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot with errorbars
    ax.errorbar(f1350_means, k_values, 
                xerr=[xerr_lower, xerr_upper],
                yerr=k_errors,
                fmt='o', ms=8, capsize=5, capthick=2,
                elinewidth=2, alpha=0.7, color='dodgerblue',
                label='Best-fit $k$ values')
    
    # Add RM ID labels
    # for i, rm_id in enumerate(rm_ids):
    #     ax.annotate(rm_id, (f1350_means[i], k_values[i]),
    #                xytext=(5, 5), textcoords='offset points',
    #                fontsize=10, alpha=0.7)
    
    ax.set_xlabel(r'$F_{\lambda,1350}~\rm [erg~cm^{-2}~s^{-1}~\AA^{-1}]$', fontsize=18)
    ax.set_ylabel(r'$k$ (metallicity gradient)', fontsize=18)
    ax.set_title('Metallicity Gradient vs Continuum Flux', fontsize=18)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, axis='both', which='major',
                   length=8, width=2, direction='in', labelsize=14)
    ax.tick_params(top=True, right=True, axis='both', which='minor',
                   length=4, width=1, direction='in')
    ax.legend(fontsize=12, loc='best')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'k_vs_f1350.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved to {output_file}")
    print(f"Plotted {len(rm_ids)} objects")
    
    # Create histogram of k distribution
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create histogram
    n, bins, patches = ax.hist(k_values, bins=50, alpha=0.7, color='dodgerblue', linewidth=1.5)
    
    # Add vertical lines for mean and median
    k_mean = k_values.mean()
    k_median = np.median(k_values)
    k_std = k_values.std()
    
    ax.axvline(k_mean, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {k_mean:.3f}')
    ax.axvline(k_median, color='green', linestyle='--', linewidth=2, 
              label=f'Median: {k_median:.3f}')
    
    # Add text box with statistics
    stats_text = f'$N = {len(k_values)}$\n'
    stats_text += f'Mean: ${k_mean:.3f} \\pm {k_std:.3f}$\n'
    stats_text += f'Median: ${k_median:.3f}$\n'
    stats_text += f'Range: $[{k_values.min():.3f}, {k_values.max():.3f}]$'
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           fontsize=12, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(r'$k$ (metallicity gradient)', fontsize=18)
    ax.set_ylabel('Number of objects', fontsize=18)
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, axis='both', which='major',
                   length=8, width=2, direction='in', labelsize=14)
    ax.tick_params(top=True, right=True, axis='both', which='minor',
                   length=4, width=1, direction='in')
    ax.legend(fontsize=12, loc='best')
    
    plt.tight_layout()
    output_file_hist = os.path.join(output_dir, 'k_distribution.png')
    plt.savefig(output_file_hist, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogram saved to {output_file_hist}")
    
    # Print summary statistics
    print(f"\nSummary statistics:")
    print(f"  k range: [{k_values.min():.3f}, {k_values.max():.3f}]")
    print(f"  Mean k: {k_mean:.3f} ± {k_std:.3f}")
    print(f"  Median k: {k_median:.3f}")
    print(f"  F1350 range: [{f1350_means.min():.2e}, {f1350_means.max():.2e}] erg cm^-2 s^-1 A^-1")


if __name__ == "__main__":
    main()
