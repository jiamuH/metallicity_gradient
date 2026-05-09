#!/usr/bin/env python3
"""
Extract line ratios from SDSS RM data for metallicity inference.

This script:
1. Loads flux data for objects that have both _c1350.dat and _si4_t.dat files
2. Aligns data by common MJD values
3. Computes line ratios: Mg II/C IV, C III]/C IV, Si IV/C IV
4. Plots ratios vs F_1350 for each object
5. Saves data for metallicity inference
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from pathlib import Path

# Set up matplotlib
plt.rcParams.update({
    'text.usetex': True,
    'axes.linewidth': 2,
    'font.family': 'serif',
    'font.weight': 'heavy',
    'font.size': 20,
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \boldmath'

# Configuration
directory = "/Users/jiamuh/sdssrm/"
output_dir = "data/alpha/observed_line_ratio_data"
plot_dir = "plots/alpha/observed_line_ratio_plots"

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# File patterns
file_patterns = {
    "mg2_flux": "_mg2_t.dat",
    "c3_flux": "_c3_t.dat",
    "c4_flux": "_c4_t.dat",
    "si4_flux": "_si4_t.dat",
    "c1350_flux": "_c1350.dat"
}


def extract_redshift_from_header(avg_w_file):
    """Extract redshift from the header of rmxxx_avg_w.dat file."""
    try:
        with open(avg_w_file, 'r') as f:
            for line in f:
                # Look for line like: #  Col 1 : Restframe Wavelength (A) for z = 1.46386
                match = re.search(r'z\s*=\s*([\d.]+)', line)
                if match:
                    return float(match.group(1))
    except Exception as e:
        print(f"Could not read header from {avg_w_file}: {e}")
    return None


def load_flux_data(rm_dir, rm_id, pattern):
    """
    Load flux data from a file, filtering out:
    1. Points where error >= 1.0 (for _t.dat files)
    2. Points with negative flux values (Col 2)
    
    For _t.dat files: Col 1 = MJD, Col 2 = Flux, Col 3 = Error bar on Col 2
    For other files: Col 1 = MJD, Col 2 = Flux
    """
    dat_file = glob.glob(f"{rm_dir}/{rm_id}{pattern}")
    if dat_file:
        try:
            data = np.loadtxt(dat_file[0], comments="#")
            if len(data) == 0:
                return None
            
            # Filter out negative flux values (Col 2)
            flux_col = data[:, 1]
            valid_mask = (flux_col > 0) & (flux_col < 1e-3) 
            
            # Filter out rows where error (Col 3) >= 1.0
            # Only apply this filter to _t.dat files (which have error column)
            if "_t.dat" in pattern and data.shape[1] >= 3:
                error_col = data[:, 2]
                # Keep rows where error < 1.0
                error_mask = error_col < 1.0
                valid_mask = valid_mask & error_mask
            
            data = data[valid_mask]
            if len(data) == 0:
                return None
            return data
        except Exception as e:
            print(f"Could not read {dat_file[0]}: {e}")
    return None


def align_flux_data(flux_dict):
    """
    Align all flux data by common MJD values.
    
    Parameters:
    -----------
    flux_dict : dict
        Dictionary with keys like 'mg2_flux', 'c3_flux', etc.
        Values are numpy arrays with shape (N, M) where first column is MJD.
        None values are allowed (missing data).
    
    Returns:
    --------
    aligned_dict : dict or None
        Dictionary with aligned flux data, or None if alignment fails.
    """
    # Filter out None values
    available_fluxes = {k: v for k, v in flux_dict.items() if v is not None}
    
    if len(available_fluxes) == 0:
        return None
    
    # Start with the first available MJD array
    first_key = list(available_fluxes.keys())[0]
    common_mjd = available_fluxes[first_key][:, 0]
    
    # Find intersection of all MJD arrays
    for key, data in available_fluxes.items():
        mjd = data[:, 0]
        common_mjd = np.intersect1d(common_mjd, mjd)
    
    if len(common_mjd) == 0:
        return None
    
    # Align all data
    aligned_dict = {}
    for key in flux_dict.keys():
        if flux_dict[key] is not None:
            mjd = flux_dict[key][:, 0]
            mask = np.isin(mjd, common_mjd)
            aligned_dict[key] = flux_dict[key][mask]
        else:
            aligned_dict[key] = None
    
    # Verify all aligned arrays have the same length
    aligned_lengths = [len(v) for v in aligned_dict.values() if v is not None]
    if len(set(aligned_lengths)) > 1:
        return None
    
    return aligned_dict


def compute_ratio_error(flux, flux_err, ref_flux, ref_flux_err):
    """Compute error for flux ratio using error propagation."""
    return np.sqrt((flux_err / ref_flux) ** 2 + (flux * ref_flux_err / ref_flux ** 2) ** 2)


def process_object(rm_dir, rm_id):
    """
    Process a single object: load data, align, compute ratios, plot, and save.
    
    Returns:
    --------
    result : dict or None
        Dictionary with processed data if successful, None otherwise.
        Dictionary contains: 'rm_id', 'redshift', 'ratios', 'ratio_errors'
    """
    # Check for avg_w.dat file to get redshift
    avg_w_file = glob.glob(f"{rm_dir}/{rm_id}_avg_w.dat")
    if not avg_w_file:
        return None
    
    # Extract redshift
    redshift = extract_redshift_from_header(avg_w_file[0])
    if redshift is None:
        return None
    
    # Check if both required files exist
    c1350_file = glob.glob(f"{rm_dir}/{rm_id}{file_patterns['c1350_flux']}")
    si4_file = glob.glob(f"{rm_dir}/{rm_id}{file_patterns['si4_flux']}")
    
    if not c1350_file or not si4_file:
        return None
    
    # Load all flux data
    flux_dict = {}
    for flux_name, pattern in file_patterns.items():
        flux_dict[flux_name] = load_flux_data(rm_dir, rm_id, pattern)
    
    # Check that we have at least c4_flux and c1350_flux (required for ratios)
    if flux_dict['c4_flux'] is None or flux_dict['c1350_flux'] is None:
        return None
    
    # Align data by common MJD
    aligned_data = align_flux_data(flux_dict)
    if aligned_data is None:
        print(f"Skipping {rm_id}: alignment failed")
        return None
    
    # Extract aligned fluxes
    c1350_data = aligned_data['c1350_flux']
    c4_data = aligned_data['c4_flux']
    
    # Get c1350 flux (column 1) and MJD (column 0)
    c1350_aligned = c1350_data[:, 1]
    mjd_aligned = c1350_data[:, 0]
    
    # Get c4 flux
    c4_aligned = c4_data[:, 1]
    
    # Filter out negative or zero fluxes
    valid_mask = (c1350_aligned > 0) & (c4_aligned > 0)
    
    # Check if we have valid data
    if np.sum(valid_mask) == 0:
        print(f"Skipping {rm_id}: no valid data points")
        return None
    
    # Apply mask
    c1350_aligned = c1350_aligned[valid_mask]
    c4_aligned = c4_aligned[valid_mask]
    mjd_aligned = mjd_aligned[valid_mask]
    
    # Initialize storage for ratios
    ratios = {}
    ratio_errors = {}
    
    # Compute Mg II/C IV ratio
    if aligned_data['mg2_flux'] is not None:
        mg2_data = aligned_data['mg2_flux']
        # Apply valid_mask first, then check for positive flux
        mg2_flux_masked = mg2_data[:, 1][valid_mask]
        mg2_err_masked = mg2_data[:, 2][valid_mask] if mg2_data.shape[1] > 2 else np.zeros_like(mg2_flux_masked)
        
        # Filter for positive mg2 flux
        mg2_mask = mg2_flux_masked > 0
        if np.sum(mg2_mask) > 0:
            ratios['mg2_c4'] = mg2_flux_masked[mg2_mask] / c4_aligned[mg2_mask]
            c4_err = (c4_data[:, 2][valid_mask][mg2_mask] 
                     if c4_data.shape[1] > 2 
                     else np.zeros_like(c4_aligned[mg2_mask]))
            ratio_errors['mg2_c4'] = compute_ratio_error(
                mg2_flux_masked[mg2_mask], mg2_err_masked[mg2_mask],
                c4_aligned[mg2_mask], c4_err
            )
            ratios['mg2_c4_c1350'] = c1350_aligned[mg2_mask]
            ratios['mg2_c4_mjd'] = mjd_aligned[mg2_mask]
    
    # Compute C III]/C IV ratio
    if aligned_data['c3_flux'] is not None:
        c3_data = aligned_data['c3_flux']
        # Apply valid_mask first, then check for positive flux
        c3_flux_masked = c3_data[:, 1][valid_mask]
        c3_err_masked = c3_data[:, 2][valid_mask] if c3_data.shape[1] > 2 else np.zeros_like(c3_flux_masked)
        
        # Filter for positive c3 flux
        c3_mask = c3_flux_masked > 0
        if np.sum(c3_mask) > 0:
            ratios['c3_c4'] = c3_flux_masked[c3_mask] / c4_aligned[c3_mask]
            c4_err = (c4_data[:, 2][valid_mask][c3_mask] 
                     if c4_data.shape[1] > 2 
                     else np.zeros_like(c4_aligned[c3_mask]))
            ratio_errors['c3_c4'] = compute_ratio_error(
                c3_flux_masked[c3_mask], c3_err_masked[c3_mask],
                c4_aligned[c3_mask], c4_err
            )
            ratios['c3_c4_c1350'] = c1350_aligned[c3_mask]
            ratios['c3_c4_mjd'] = mjd_aligned[c3_mask]
    
    # Compute Si IV/C IV ratio
    if aligned_data['si4_flux'] is not None:
        si4_data = aligned_data['si4_flux']
        # Apply valid_mask first, then check for positive flux
        si4_flux_masked = si4_data[:, 1][valid_mask]
        si4_err_masked = si4_data[:, 2][valid_mask] if si4_data.shape[1] > 2 else np.zeros_like(si4_flux_masked)
        
        # Filter for positive si4 flux
        si4_mask = si4_flux_masked > 0
        if np.sum(si4_mask) > 0:
            ratios['si4_c4'] = si4_flux_masked[si4_mask] / c4_aligned[si4_mask]
            c4_err = (c4_data[:, 2][valid_mask][si4_mask] 
                     if c4_data.shape[1] > 2 
                     else np.zeros_like(c4_aligned[si4_mask]))
            ratio_errors['si4_c4'] = compute_ratio_error(
                si4_flux_masked[si4_mask], si4_err_masked[si4_mask],
                c4_aligned[si4_mask], c4_err
            )
            ratios['si4_c4_c1350'] = c1350_aligned[si4_mask]
            ratios['si4_c4_mjd'] = mjd_aligned[si4_mask]
    
    # Check if we have any ratios
    if len(ratios) == 0:
        print(f"Skipping {rm_id}: no valid ratios computed")
        return None
    
    # Plot ratios
    plot_ratios(rm_id, ratios, ratio_errors, redshift)
    
    # Save data
    save_ratios(rm_id, ratios, ratio_errors, redshift)
    
    # Return data for combined plots
    return {
        'rm_id': rm_id,
        'redshift': redshift,
        'ratios': ratios,
        'ratio_errors': ratio_errors
    }


def plot_ratios(rm_id, ratios, ratio_errors, redshift):
    """Plot line ratios vs F_1350 for a single object."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot Mg II/C IV
    if 'mg2_c4' in ratios:
        c1350 = ratios['mg2_c4_c1350']
        ratio = ratios['mg2_c4']
        err = ratio_errors.get('mg2_c4', None)
        if err is not None and len(err) == len(ratio):
            ax.errorbar(c1350, ratio, yerr=err, fmt='o', ms=5, 
                       label='Mg II/C IV', alpha=0.7, capsize=3)
        else:
            ax.scatter(c1350, ratio, s=20, label='Mg II/C IV', alpha=0.7)
    
    # Plot C III]/C IV
    if 'c3_c4' in ratios:
        c1350 = ratios['c3_c4_c1350']
        ratio = ratios['c3_c4']
        err = ratio_errors.get('c3_c4', None)
        if err is not None and len(err) == len(ratio):
            ax.errorbar(c1350, ratio, yerr=err, fmt='s', ms=5,
                       label='C III]/C IV', alpha=0.7, capsize=3)
        else:
            ax.scatter(c1350, ratio, s=20, label='C III]/C IV', alpha=0.7)
    
    # Plot Si IV/C IV
    if 'si4_c4' in ratios:
        c1350 = ratios['si4_c4_c1350']
        ratio = ratios['si4_c4']
        err = ratio_errors.get('si4_c4', None)
        if err is not None and len(err) == len(ratio):
            ax.errorbar(c1350, ratio, yerr=err, fmt='^', ms=5,
                       label='Si IV/C IV', alpha=0.7, capsize=3)
        else:
            ax.scatter(c1350, ratio, s=20, label='Si IV/C IV', alpha=0.7)
    
    # Customize plot
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$F_{\lambda,1350}~\rm [erg~cm^{-2}~s^{-1}~\AA^{-1}]$')
    ax.set_ylabel('Line Ratio')
    ax.set_title(f'{rm_id} (z = {redshift:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True, 
                   length=9, width=2, direction='in')
    ax.tick_params(axis='both', which='minor', top=True, right=True, 
                   length=4, width=2, direction='in')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/{rm_id}_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_ratios(rm_id, ratios, ratio_errors, redshift):
    """Save line ratios to file for metallicity inference."""
    output_file = f'{output_dir}/{rm_id}_line_ratios.dat'
    
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"# Line ratios for {rm_id} (z = {redshift:.6f})\n")
        f.write("# Format: F_1350  Ratio  Ratio_Error  MJD  Ratio_Type\n")
        f.write("# Ratio types: mg2_c4, c3_c4, si4_c4\n")
        f.write("# F_1350 units: erg cm^-2 s^-1 A^-1\n")
        f.write("#\n")
        
        # Write Mg II/C IV
        if 'mg2_c4' in ratios:
            c1350 = ratios['mg2_c4_c1350']
            ratio = ratios['mg2_c4']
            mjd = ratios['mg2_c4_mjd']
            err = ratio_errors.get('mg2_c4', np.zeros_like(ratio))
            for f1350, r, r_err, m in zip(c1350, ratio, err, mjd):
                f.write(f"{f1350:.18e}  {r:.18e}  {r_err:.18e}  {m:.6f}  mg2_c4\n")
        
        # Write C III]/C IV
        if 'c3_c4' in ratios:
            c1350 = ratios['c3_c4_c1350']
            ratio = ratios['c3_c4']
            mjd = ratios['c3_c4_mjd']
            err = ratio_errors.get('c3_c4', np.zeros_like(ratio))
            for f1350, r, r_err, m in zip(c1350, ratio, err, mjd):
                f.write(f"{f1350:.18e}  {r:.18e}  {r_err:.18e}  {m:.6f}  c3_c4\n")
        
        # Write Si IV/C IV
        if 'si4_c4' in ratios:
            c1350 = ratios['si4_c4_c1350']
            ratio = ratios['si4_c4']
            mjd = ratios['si4_c4_mjd']
            err = ratio_errors.get('si4_c4', np.zeros_like(ratio))
            for f1350, r, r_err, m in zip(c1350, ratio, err, mjd):
                f.write(f"{f1350:.18e}  {r:.18e}  {r_err:.18e}  {m:.6f}  si4_c4\n")
    
    print(f"Saved data for {rm_id} to {output_file}")


def plot_combined_ratios(all_data, ratio_type):
    """
    Create combined plot for a single line ratio type showing all objects.
    
    Parameters:
    -----------
    all_data : list
        List of dictionaries with processed object data
    ratio_type : str
        One of 'mg2_c4', 'c3_c4', 'si4_c4'
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Collect data for this ratio type
    for obj_data in all_data:
        rm_id = obj_data['rm_id']
        ratios = obj_data['ratios']
        ratio_errors = obj_data['ratio_errors']
        
        if ratio_type not in ratios:
            continue
        
        c1350 = ratios[f'{ratio_type}_c1350']
        ratio = ratios[ratio_type]
        err = ratio_errors.get(ratio_type, None)
        
        # Plot scatter points
        if err is not None and len(err) == len(ratio):
            ax.errorbar(c1350, ratio, yerr=err, fmt='o', ms=3, 
                       alpha=0.5, capsize=1, elinewidth=0.5)
        else:
            ax.scatter(c1350, ratio, s=10, alpha=0.5)
        
        # Calculate center of scatter cloud for label
        if len(c1350) > 0:
            center_x = np.median(c1350)
            center_y = np.median(ratio)
            # Add RM number label at center
            ax.text(center_x, center_y, rm_id, 
                   ha='center', va='center',
                   fontsize=8, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='white', 
                            edgecolor='black', 
                            alpha=0.7))
    
    # Set labels based on ratio type
    ratio_labels = {
        'mg2_c4': 'Mg II/C IV',
        'c3_c4': 'C III]/C IV',
        'si4_c4': 'Si IV/C IV'
    }
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$F_{\lambda,1350}~\rm [erg~cm^{-2}~s^{-1}~\AA^{-1}]$', fontsize=16)
    ax.set_ylabel(f'{ratio_labels[ratio_type]}', fontsize=16)
    ax.set_title(f'Combined {ratio_labels[ratio_type]} vs F_1350', fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True, 
                   length=9, width=2, direction='in', labelsize=12)
    ax.tick_params(axis='both', which='minor', top=True, right=True, 
                   length=4, width=2, direction='in')
    
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/combined_{ratio_type}_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot: {plot_dir}/combined_{ratio_type}_ratios.png")


def main():
    """Main processing loop."""
    # Get list of all rm directories
    rm_directories = sorted(glob.glob(f"{directory}/rm*"))
    
    print(f"Found {len(rm_directories)} RM directories")
    print(f"Processing objects with both _c1350.dat and _si4_t.dat files...")
    print("Filtering out:")
    print("  - Data points where error >= 1.0")
    print("  - Data points with negative flux values")
    print()
    
    processed_count = 0
    skipped_count = 0
    all_data = []  # Store all processed data for combined plots
    
    for rm_dir in rm_directories:
        rm_id = rm_dir.split("/")[-1]
        
        result = process_object(rm_dir, rm_id)
        if result is not None:
            processed_count += 1
            all_data.append(result)
        else:
            skipped_count += 1
    
    print()
    print(f"Processing complete!")
    print(f"  Processed: {processed_count} objects")
    print(f"  Skipped: {skipped_count} objects")
    print(f"  Data saved to: {output_dir}/")
    print(f"  Individual plots saved to: {plot_dir}/")
    
    # Create combined plots for each ratio type
    if len(all_data) > 0:
        print()
        print("Creating combined plots...")
        for ratio_type in ['mg2_c4', 'c3_c4', 'si4_c4']:
            plot_combined_ratios(all_data, ratio_type)
        print(f"  Combined plots saved to: {plot_dir}/")


if __name__ == "__main__":
    main()
