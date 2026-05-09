"""
Script to compute and plot line ratios (MgII/CIV, (SiIV+OIV)/CIV) 
as a function of continuum luminosity with breathing effect.

The breathing effect: r ∝ Q^(1/2), so as Q changes, the BLR radius changes,
and the ionizing flux φ = Q/(4πr²) changes accordingly.

Usage:
    python line_ratio_breathing_effect.py

Output:
    - Plots saved in 'plots/alpha/line_ratio_plots' directory
    - Summary plot saved as 'line_ratios_summary_rref_log*.png'
    
The script:
1. Loads Cloudy model data (3D grid: n, phi, Z)
2. Collapses over density axis (marginalizes)
3. Creates 2D interpolation functions for (phi, Z)
4. Implements breathing effect with Gaussian window function
5. Computes line ratios as function of Q for different Z(r) profiles
6. Generates plots showing how ratios change with luminosity
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import os
from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

# ============================================================================
# Configuration
# ============================================================================

# Grid definitions
Z_grid = np.arange(1, 20.5, 0.5)          # 39 metallicity values (1 to 20 Z_sun)
phi_grid = np.arange(17, 21.5, 0.5)       # 9 φ values
n_grid = np.arange(9, 12.5, 0.5)          # 7 n values

# File path
basepath = '/Users/jiamuh/c23.01/my_models/loc_metal'
model_file = os.path.join(basepath, 'strong_LOC_varym_N25_LineList_BLR_Fe2.txt')

# Line names mapping
line_names = {
    'Mg2': 'blnd 2798.00A',
    'C4':  'blnd 1549.00A',
    'Si4': 'BLND 1397.00A',
    'O4':  'blnd 1402.00A',
}

# Breathing effect parameters
Q_ref = 1e56  # Reference ionizing photon rate
logQ_min, logQ_max, nQ = 54, 58, 50  # Q grid for plotting
logQ_plot = np.linspace(logQ_min, logQ_max, nQ)
Q_plot = 10**logQ_plot

# Radius range for integration
log_rin, log_rout = 17.0, 20.5
r = np.logspace(log_rin, log_rout, 300)  # Fine r grid for integration

# Window function parameters
sigma = 0.2  # Gaussian width in log r
# Fixed gamma for MCMC (Korista & Goad 2019 responsivity power-law)
gamma_mcmc = -1.2
# For plotting: use a subset
gammas_plot = [-1.0, -1.2]  # Power-law indices for flux weighting (for example plots)

# Breathing factor β: parameterizes how the BLR responds to luminosity changes
# r_center = r_ref * (Q/Q_ref)^(0.5*β)
# β = 0: no breathing (r fixed, φ increases with Q)
# β = 1: full breathing (r ∝ Q^0.5, φ constant)
# β < 1: partial breathing (r increases with Q but slower, φ still increases)
breathing_factors_plot = [0.0, 0.5, 1.0]  # No, partial, full breathing
# Beta grid for MCMC: 11 values from 0.0 to 1.0
beta_grid_mcmc = np.arange(0.0, 1.05, 0.1)  # [0.0, 0.1, 0.2, ..., 1.0]
# Use LaTeX-safe labels (no Unicode characters)
breathing_labels = {0.0: 'No breathing', 0.5: 'Partial', 1.0: 'Full breathing'}

# Reference radius grid (for different r_ref values)
# Compute valid r_ref range considering BOTH φ and r constraints across all Q values
#
# Key insight: r_center = r_ref * (Q / Q_ref)^0.5 varies with Q
# But φ_center = log(Q_ref) - 2*log(r_ref) is CONSTANT (independent of Q) - the breathing effect!
#
# Constraints:
# 1. φ constraint: φ_center ± 2σ must be within [φ_min, φ_max]
# 2. r constraint: r_center(Q) must be within [r_in, r_out] for all Q in [Q_min, Q_max]
# 3. (Optional, stricter) ±1σ extent in r within [r_in, r_out]

phi_min, phi_max = phi_grid.min(), phi_grid.max()
log_Q_ref = np.log10(Q_ref)
log_r_in, log_r_out = np.log10(r.min()), np.log10(r.max())

print(f"\n{'='*70}")
print("Computing valid r_ref range (accounting for Q variation)")
print(f"{'='*70}")
print(f"r grid: log r ∈ [{log_r_in:.2f}, {log_r_out:.2f}]")
print(f"φ grid: φ ∈ [{phi_min:.1f}, {phi_max:.1f}]")
print(f"Q range: log Q ∈ [{logQ_min}, {logQ_max}]")
print(f"Q_ref: log Q_ref = {log_Q_ref:.1f}")
print(f"σ (window width in log r): {sigma}")

# --- Constraint 1: φ constraint ---
# φ_center ± 2σ ⊆ [φ_min, φ_max]
# → φ_min + 2σ ≤ φ_center ≤ φ_max - 2σ
phi_center_min_from_phi = phi_min + 2 * sigma
phi_center_max_from_phi = phi_max - 2 * sigma
# φ_center = log(Q_ref) - 2*log(r_ref) → log(r_ref) = (log(Q_ref) - φ_center) / 2
log_r_ref_max_from_phi = (log_Q_ref - phi_center_min_from_phi) / 2
log_r_ref_min_from_phi = (log_Q_ref - phi_center_max_from_phi) / 2

print(f"\n1. φ constraint (±1σ in φ within grid):")
print(f"   φ_center must be in [{phi_center_min_from_phi:.1f}, {phi_center_max_from_phi:.1f}]")
print(f"   → log r_ref ∈ [{log_r_ref_min_from_phi:.2f}, {log_r_ref_max_from_phi:.2f}]")
print(f"   Width: {log_r_ref_max_from_phi - log_r_ref_min_from_phi:.2f} dex")

# --- Constraint 2: r constraint (r_center within r grid for all Q) ---
# r_center(Q) = r_ref * (Q / Q_ref)^0.5
# At Q_min: r_center = r_ref * 10^((logQ_min - log_Q_ref)/2)
# At Q_max: r_center = r_ref * 10^((logQ_max - log_Q_ref)/2)
delta_log_r_at_Qmin = (logQ_min - log_Q_ref) / 2  # e.g., (54 - 56)/2 = -1
delta_log_r_at_Qmax = (logQ_max - log_Q_ref) / 2  # e.g., (56 - 56)/2 = 0

# For r_center to be within [r_in, r_out]:
# At Q_min: log(r_ref) + delta_log_r_at_Qmin ≥ log_r_in → log(r_ref) ≥ log_r_in - delta_log_r_at_Qmin
# At Q_min: log(r_ref) + delta_log_r_at_Qmin ≤ log_r_out → log(r_ref) ≤ log_r_out - delta_log_r_at_Qmin
# At Q_max: log(r_ref) + delta_log_r_at_Qmax ≥ log_r_in → log(r_ref) ≥ log_r_in - delta_log_r_at_Qmax
# At Q_max: log(r_ref) + delta_log_r_at_Qmax ≤ log_r_out → log(r_ref) ≤ log_r_out - delta_log_r_at_Qmax
log_r_ref_min_from_r = max(log_r_in - delta_log_r_at_Qmin, log_r_in - delta_log_r_at_Qmax)
log_r_ref_max_from_r = min(log_r_out - delta_log_r_at_Qmin, log_r_out - delta_log_r_at_Qmax)

print(f"\n2. r constraint (r_center within r grid for all Q):")
print(f"   At Q_min=10^{logQ_min}: r_center = r_ref × 10^{delta_log_r_at_Qmin:.1f}")
print(f"   At Q_max=10^{logQ_max}: r_center = r_ref × 10^{delta_log_r_at_Qmax:.1f}")
print(f"   → log r_ref ∈ [{log_r_ref_min_from_r:.2f}, {log_r_ref_max_from_r:.2f}]")
print(f"   Width: {log_r_ref_max_from_r - log_r_ref_min_from_r:.2f} dex")

# --- Constraint 3 (stricter): ±1σ extent in r within r grid for all Q ---
# r_low(Q) = r_center(Q) / 10^σ, r_high(Q) = r_center(Q) * 10^σ
# At Q_min: log(r_low) = log(r_ref) + delta_log_r_at_Qmin - σ ≥ log_r_in
# At Q_min: log(r_high) = log(r_ref) + delta_log_r_at_Qmin + σ ≤ log_r_out
# At Q_max: log(r_low) = log(r_ref) + delta_log_r_at_Qmax - σ ≥ log_r_in
# At Q_max: log(r_high) = log(r_ref) + delta_log_r_at_Qmax + σ ≤ log_r_out
log_r_ref_min_from_r_1sig = max(log_r_in - delta_log_r_at_Qmin + sigma, 
                                 log_r_in - delta_log_r_at_Qmax + sigma)
log_r_ref_max_from_r_1sig = min(log_r_out - delta_log_r_at_Qmin - sigma,
                                 log_r_out - delta_log_r_at_Qmax - sigma)

print(f"\n3. r constraint (±1σ extent within r grid for all Q) [stricter]:")
print(f"   → log r_ref ∈ [{log_r_ref_min_from_r_1sig:.2f}, {log_r_ref_max_from_r_1sig:.2f}]")
print(f"   Width: {log_r_ref_max_from_r_1sig - log_r_ref_min_from_r_1sig:.2f} dex")

# --- Combined constraints ---
# Use constraint 1 (φ) AND constraint 2 (r_center within r grid)
log_r_ref_min_combined = max(log_r_ref_min_from_phi, log_r_ref_min_from_r)
log_r_ref_max_combined = min(log_r_ref_max_from_phi, log_r_ref_max_from_r)
combined_width = log_r_ref_max_combined - log_r_ref_min_combined

print(f"\n4. Combined (φ + r_center constraints):")
print(f"   → log r_ref ∈ [{log_r_ref_min_combined:.2f}, {log_r_ref_max_combined:.2f}]")
print(f"   Width: {combined_width:.2f} dex")

# Select r_ref values
if combined_width < 0:
    print(f"\n   WARNING: No valid r_ref range exists!")
    print(f"   Using best compromise (middle of r grid with φ constraint)...")
    log_r_ref_mid = (log_Q_ref - (phi_min + phi_max) / 2) / 2
    r_ref_grid = np.array([10**log_r_ref_mid])
elif combined_width < 0.3:
    print(f"\n   NOTE: Valid range is narrow. Using 3 values within range.")
    log_r_ref_values = np.linspace(log_r_ref_min_combined, log_r_ref_max_combined, 3)
    r_ref_grid = 10**log_r_ref_values
else:
    n_r_ref = 5
    log_r_ref_values = np.linspace(log_r_ref_min_combined, log_r_ref_max_combined, n_r_ref)
    r_ref_grid = 10**log_r_ref_values

print(f"\n   Selected log r_ref values: {np.log10(r_ref_grid)}")
print(f"   Corresponding φ_center values: {log_Q_ref - 2*np.log10(r_ref_grid)}")
print(f"{'='*70}\n")

# Metallicity gradient profiles
# For plotting: use a subset of k values
k_values_plot = [-0.5, 0.0, 0.5]  # Power-law indices for Z(r) - for plotting
# For MCMC: full grid of k values (narrowed range for physical metallicity gradients)
k_min, k_max, n_k = -0.5, 0.5, 50
k_grid = np.linspace(k_min, k_max, n_k)  # Full grid for MCMC inference

Z_min, Z_max = min(Z_grid), max(Z_grid)
cm_to_pc = 3.086e18

# Plotting setup
plt.rcParams.update({
    'text.usetex': True,
    'axes.linewidth': 2,
    'font.family': 'serif',
    'font.weight': 'heavy',
    'font.size': 16,
})

# Use plasma colormap for k values (matching fit_line_ratios.py)
k_min_plot = min(k_values_plot)
k_max_plot = max(k_values_plot)
norm_plot = plt.Normalize(vmin=k_min_plot, vmax=k_max_plot)
cmap_plot = plt.colormaps['plasma']

# Generate colors and labels for k_values_plot
colors_gradient = [cmap_plot(norm_plot(k)) for k in k_values_plot]
labels_gradient = [rf"$k={k:.1f}$" for k in k_values_plot]

# Custom colormap
colors_cmap = [
    (0.15, 0.08, 0.40),
    (0.294, 0.161, 0.569),
    (0.529, 0.173, 0.635),
    (0.753, 0.212, 0.616),
    (0.918, 0.310, 0.533),
    (0.980, 0.471, 0.463),
    (0.965, 0.663, 0.478),
    (0.929, 0.851, 0.639),
]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors_cmap, N=256)

# ============================================================================
# Load and process data
# ============================================================================

print("Loading model data...")
# comment='#' skips GRID_DELIMIT lines but also the header line (#lineslist...)
# So read without comment, then filter
df = pd.read_csv(model_file, sep='\t', header=0)
# Drop rows where the first column contains GRID_DELIMIT markers
df = df[~df.iloc[:, 0].astype(str).str.contains('GRID_DELIMIT')].reset_index(drop=True)
print(f"  Loaded {len(df)} rows (expected {len(Z_grid) * len(phi_grid) * len(n_grid)})")
assert len(df) == len(Z_grid) * len(phi_grid) * len(n_grid), \
    f"Row count mismatch! Got {len(df)}, expected {len(Z_grid) * len(phi_grid) * len(n_grid)}"

# Initialize 3D arrays: shape = (len(n), len(phi), len(Z))
shape = (len(n_grid), len(phi_grid), len(Z_grid))
intensity_data = {key: np.zeros(shape) for key in line_names}

# Fill 3D arrays
for j_phi, phi_val in enumerate(phi_grid):
    for i_n, n_val in enumerate(n_grid):
        index = j_phi * len(n_grid) + i_n
        start = index * len(Z_grid)
        end = start + len(Z_grid)
        df_slice = df.iloc[start:end]
        for key, colname in line_names.items():
            intensity_data[key][i_n, j_phi, :] = df_slice[colname].values

print("Collapsing over density axis...")
# Collapse over density (marginalize): average along axis=0
collapsed_data = {}
for key in line_names:
    collapsed_data[key] = np.mean(intensity_data[key], axis=0)  # Shape: (len(phi), len(Z))

# RectBivariateSpline(x, y, z) expects z with shape (len(x), len(y))
# So for RectBivariateSpline(phi_grid, Z_grid, z), z should be (len(phi_grid), len(Z_grid))
# collapsed_data already has shape (len(phi), len(Z)) = (9, 29), which is correct!
c4_avgs = collapsed_data['C4']  # Shape: (len(phi), len(Z)) = (9, 29)
mg2_avgs = collapsed_data['Mg2']
si4_avgs = collapsed_data['Si4']
o4_avgs = collapsed_data['O4']

# Verify shapes
print(f"phi_grid shape: {phi_grid.shape}, Z_grid shape: {Z_grid.shape}")
print(f"c4_avgs shape: {c4_avgs.shape} (expected: ({len(phi_grid)}, {len(Z_grid)}))")

print("Creating 2D interpolation functions...")
# Create 2D interpolation functions over (phi, Z)
# Note: RectBivariateSpline(x, y, z) where z.shape = (len(x), len(y))
C4_interp_2D = RectBivariateSpline(phi_grid, Z_grid, c4_avgs, kx=2, ky=2)
Mg2_interp_2D = RectBivariateSpline(phi_grid, Z_grid, mg2_avgs, kx=2, ky=2)
Si4_interp_2D = RectBivariateSpline(phi_grid, Z_grid, si4_avgs, kx=2, ky=2)
O4_interp_2D = RectBivariateSpline(phi_grid, Z_grid, o4_avgs, kx=2, ky=2)

# Verify interpolation: check that MgII/CIV increases with Z
print("\nVerifying interpolation against original data...")
phi_test = phi_grid[len(phi_grid)//2]  # Middle phi value
print(f"Testing at phi = {phi_test:.2f}:")
# Check original data
idx_phi = np.where(phi_grid == phi_test)[0][0]
mg2c4_original = mg2_avgs[idx_phi, :] / c4_avgs[idx_phi, :]
print(f"  Original data: Z={Z_grid[0]:.1f} -> MgII/CIV={mg2c4_original[0]:.4f}")
print(f"                  Z={Z_grid[-1]:.1f} -> MgII/CIV={mg2c4_original[-1]:.4f}")
# Check interpolation (grid=False returns scalar, not array)
mg2_interp_low = float(Mg2_interp_2D(phi_test, Z_grid[0], grid=False))
c4_interp_low = float(C4_interp_2D(phi_test, Z_grid[0], grid=False))
mg2_interp_high = float(Mg2_interp_2D(phi_test, Z_grid[-1], grid=False))
c4_interp_high = float(C4_interp_2D(phi_test, Z_grid[-1], grid=False))
mg2c4_interp_low = mg2_interp_low / c4_interp_low
mg2c4_interp_high = mg2_interp_high / c4_interp_high
print(f"  Interpolation: Z={Z_grid[0]:.1f} -> MgII/CIV={mg2c4_interp_low:.4f}")
print(f"                  Z={Z_grid[-1]:.1f} -> MgII/CIV={mg2c4_interp_high:.4f}")
if mg2c4_interp_high > mg2c4_interp_low:
    print("  ✓ Interpolation is correct: ratio increases with Z")
else:
    print("  ⚠ WARNING: Interpolation shows decreasing ratio with Z (may indicate issue)")

# ============================================================================
# Define metallicity profiles Z(r)
# ============================================================================

def create_Z_profile(r, k, Z_min, Z_max):
    """
    Create metallicity profile Z(r) with power-law index k.
    
    Model: Z(r) ∝ r^k, normalized to span [Z_min, Z_max]
    """
    if k == 0:
        # Constant Z: use average
        return np.full_like(r, (Z_min + Z_max) / 2.0)
    else:
        # Z(r) ∝ r^k
        Z_unscaled = r**k
        # Normalize to span [Z_min, Z_max]
        Z_scaled = (Z_unscaled - np.min(Z_unscaled)) / (np.max(Z_unscaled) - np.min(Z_unscaled))
        Z_scaled = Z_scaled * (Z_max - Z_min) + Z_min
        return Z_scaled

# Create Z profiles for plotting (subset of k values)
Z_profiles = []
for k, color, label in zip(k_values_plot, colors_gradient, labels_gradient):
    Z_r = create_Z_profile(r, k, Z_min, Z_max)
    Z_profiles.append((label, color, Z_r))

# ============================================================================
# Plot Z(r) profiles first
# ============================================================================

print("\nPlotting Z(r) profiles...")
fig, ax = plt.subplots(figsize=(10, 7))

r_pc = r / cm_to_pc  # Convert to parsecs for plotting

for label, color, Z_r in Z_profiles:
    ax.plot(np.log10(r_pc), Z_r, color=color, linestyle='-', linewidth=2.5, label=label)

ax.set_xlabel(r'$\log r\ \mathrm{[pc]}$', fontsize=18)
ax.set_ylabel(r'$Z\ [{\rm Z}_\odot]$', fontsize=18)
ax.set_title(r'Metallicity Profiles: $Z(r) \propto r^k$', fontsize=18)
ax.minorticks_on()
ax.tick_params(top=True, right=True, axis='both', which='major', 
               length=8, width=2, direction='in', labelsize=14)
ax.tick_params(top=True, right=True, axis='both', which='minor', 
               length=4, width=1, direction='in')
ax.legend(fontsize=12, loc='best', framealpha=0.9)

plt.tight_layout()
plt.savefig('Z_profiles_vs_r.png', dpi=300, bbox_inches='tight')
plt.close()

print("Z(r) profiles plot saved as 'Z_profiles_vs_r.png'")

# ============================================================================
# Window function for breathing effect
# ============================================================================

def gaussian_powerlaw_weight(r, r_center, gamma=-1.0, sigma=1.0):
    """
    Gaussian window function with power-law flux weighting.
    
    W(r; Q) = (1 / (sqrt(2π) σ r² ln 10)) * exp(-(log r - log r_Q)² / (2σ²))
    with additional r^gamma factor for flux weighting.
    """
    log_r = np.log10(r)
    log_r_center = np.log10(r_center)
    weight = np.exp(-0.5 * ((log_r - log_r_center) / sigma)**2) * r**gamma
    # Normalize so integral = 1
    norm = np.trapezoid(weight, r)
    if norm > 0:
        weight /= norm
    return weight

# ============================================================================
# Compute line ratios as function of Q with breathing effect
# ============================================================================

def compute_line_ratios_vs_Q(r_ref, Z_profile, gamma, breathing_factor=1.0):
    """
    Compute line ratios as a function of Q for given r_ref, Z(r), and gamma.
    
    Parameters:
    -----------
    r_ref : float
        Reference radius (cm) at Q_ref
    Z_profile : array
        Z(r) profile
    gamma : float
        Power-law index for flux weighting
    breathing_factor : float
        Breathing response parameter β ∈ [0, 1]:
        - β = 0: no breathing (r_center fixed at r_ref)
        - β = 1: full breathing (r_center ∝ Q^0.5, maintaining constant φ)
        - 0 < β < 1: partial breathing (r_center ∝ Q^(0.5β))
        
    Returns:
    --------
    results : dict
        Dictionary with logQ, MgII/CIV, (SiIV+OIV)/CIV ratios
    """
    civ_list, mg2_list, si4_list, o4_list = [], [], [], []
    
    for Q in Q_plot:
        # Breathing effect: r_center = r_ref * (Q/Q_ref)^(0.5 * β)
        # β = 0: r_center = r_ref (no breathing)
        # β = 1: r_center ∝ Q^0.5 (full breathing, constant φ)
        r_center = r_ref * (Q / Q_ref)**(0.5 * breathing_factor)
        
        # Compute phi at each r: log10(phi) = log10(Q) - 2*log10(r)
        phi_r = np.log10(Q) - 2 * np.log10(r)
        # Clip to interpolation range to avoid extrapolation (phi < 17 or > 21.5 gives wrong values)
        phi_r_clip = np.clip(phi_r, phi_grid.min(), phi_grid.max())
        
        # Get window weights
        weights = gaussian_powerlaw_weight(r, r_center, gamma=gamma, sigma=sigma)
        
        # Interpolate line intensities at (phi_r, Z_r)
        # grid=False returns scalar for each point, so we loop through
        CIV = np.array([float(C4_interp_2D(phi_r_clip[i], Z_profile[i], grid=False)) 
                       for i in range(len(phi_r))])
        MgII = np.array([float(Mg2_interp_2D(phi_r_clip[i], Z_profile[i], grid=False)) 
                        for i in range(len(phi_r))])
        SiIV = np.array([float(Si4_interp_2D(phi_r_clip[i], Z_profile[i], grid=False)) 
                        for i in range(len(phi_r))])
        OIV = np.array([float(O4_interp_2D(phi_r_clip[i], Z_profile[i], grid=False)) 
                       for i in range(len(phi_r))])
        
        # Weighted integral over r
        civ_list.append(np.trapezoid(CIV * weights, r))
        mg2_list.append(np.trapezoid(MgII * weights, r))
        si4_list.append(np.trapezoid(SiIV * weights, r))
        o4_list.append(np.trapezoid(OIV * weights, r))
    
    civ_arr = np.array(civ_list)
    mg2_arr = np.array(mg2_list)
    si4_arr = np.array(si4_list)
    o4_arr = np.array(o4_list)
    
    return {
        'logQ': logQ_plot,
        'CIV': civ_arr,
        'MgII': mg2_arr,
        'SiIV': si4_arr,
        'OIV': o4_arr,
        'MgII/CIV': mg2_arr / civ_arr,
        '(SiIV+OIV)/CIV': (si4_arr + o4_arr) / civ_arr
    }

# ============================================================================
# Compute results for all combinations
# ============================================================================

print("Computing line ratios with breathing effect...")
print(f"  Breathing factors: {breathing_factors_plot}")
all_results = {}

for r_ref in r_ref_grid:
    results_breathing = {}
    for beta in breathing_factors_plot:
        results_gamma = {}
        for gamma in gammas_plot:
            results_profile = {}
            for label, color, Z_r in Z_profiles:
                results = compute_line_ratios_vs_Q(r_ref, Z_r, gamma, breathing_factor=beta)
                results['color'] = color
                results_profile[label] = results
            results_gamma[gamma] = results_profile
        results_breathing[beta] = results_gamma
    all_results[r_ref] = results_breathing

# ============================================================================
# Plotting functions
# ============================================================================

def Q_to_log_rQ(logQ, r_ref, breathing_factor=1.0):
    """Convert log Q to log r_Q in pc."""
    Q = 10**logQ
    rQ = r_ref * (Q / Q_ref)**(0.5 * breathing_factor)
    return np.log10(rQ / cm_to_pc)

def log_rQ_to_logQ(log_r_pc, r_ref, breathing_factor=1.0):
    """Convert log r_Q in pc to log Q."""
    r_pc = 10**log_r_pc
    r = r_pc * cm_to_pc
    if breathing_factor == 0:
        return np.nan  # r doesn't change with Q when β=0
    Q = Q_ref * (r / r_ref)**(2.0 / breathing_factor)
    return np.log10(Q)

def plot_line_ratios_vs_Q(r_ref, breathing_factor, output_dir="plots/alpha/line_ratio_plots"):
    """Plot line ratios vs Q for a given r_ref and breathing factor."""
    os.makedirs(output_dir, exist_ok=True)
    results_gamma = all_results[r_ref][breathing_factor]
    gamma1, gamma2 = gammas_plot
    log_rref = np.log10(r_ref)
    beta_str = f"beta{breathing_factor:.1f}"
    
    # Plot 1: MgII/CIV
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for label in results_gamma[gamma1]:
        color = results_gamma[gamma1][label]['color']
        logQ = results_gamma[gamma1][label]['logQ']
        ratio_1 = results_gamma[gamma1][label]['MgII/CIV']
        ratio_2 = results_gamma[gamma2][label]['MgII/CIV']
        
        ax.fill_between(logQ, np.minimum(ratio_1, ratio_2),
                       np.maximum(ratio_1, ratio_2),
                       color=color, alpha=0.6, label=label)
    
    # Top axis: log10(r_Q / pc) - only makes sense for β > 0
    if breathing_factor > 0:
        secax = ax.secondary_xaxis('top', functions=(
            lambda x, r=r_ref, b=breathing_factor: Q_to_log_rQ(x, r, b),
            lambda x, r=r_ref, b=breathing_factor: log_rQ_to_logQ(x, r, b)
        ))
        secax.set_xlabel(r'$\log\, r_Q\ \mathrm{[pc]}$', labelpad=10)
    
    ax.set_xlabel(r'$\log Q$')
    ax.set_ylabel(r'$\rm MgII / CIV$')
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, axis='both', which='major', 
                   length=8, width=1.5, direction='in')
    ax.tick_params(top=True, right=True, axis='both', which='minor', 
                   length=4, width=1, direction='in')
    ax.legend(title=r"$\rm Z\ Profile$", fontsize=11, loc='upper right')
    ax.set_title(rf'{breathing_labels[breathing_factor]} ($\beta={breathing_factor}$)')
    
    fname = f"MgII_CIV_ratio_rref_log{log_rref:.2f}_{beta_str}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: (SiIV+OIV)/CIV
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for label in results_gamma[gamma1]:
        color = results_gamma[gamma1][label]['color']
        logQ = results_gamma[gamma1][label]['logQ']
        ratio_1 = results_gamma[gamma1][label]['(SiIV+OIV)/CIV']
        ratio_2 = results_gamma[gamma2][label]['(SiIV+OIV)/CIV']
        
        ax.fill_between(logQ, np.minimum(ratio_1, ratio_2),
                       np.maximum(ratio_1, ratio_2),
                       color=color, alpha=0.6, label=label)
    
    # Top axis
    if breathing_factor > 0:
        secax = ax.secondary_xaxis('top', functions=(
            lambda x, r=r_ref, b=breathing_factor: Q_to_log_rQ(x, r, b),
            lambda x, r=r_ref, b=breathing_factor: log_rQ_to_logQ(x, r, b)
        ))
        secax.set_xlabel(r'$\log\, r_Q\ \mathrm{[pc]}$', labelpad=10)
    
    ax.set_xlabel(r'$\log Q$')
    ax.set_ylabel(r'$\rm (SiIV + OIV) / CIV$')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, axis='both', which='major', 
                   length=8, width=1.5, direction='in')
    ax.tick_params(top=True, right=True, axis='both', which='minor', 
                   length=4, width=1, direction='in')
    ax.legend(title=r"$\rm Z\ Profile$", fontsize=11, loc='upper right')
    ax.set_title(rf'{breathing_labels[breathing_factor]} ($\beta={breathing_factor}$)')
    
    fname = f"SiIV_OIV_CIV_ratio_rref_log{log_rref:.2f}_{beta_str}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# Generate all plots
# ============================================================================

print("Generating plots...")
for r_ref in r_ref_grid:
    for beta in breathing_factors_plot:
        plot_line_ratios_vs_Q(r_ref, beta)

print("Done! Plots saved in 'plots/alpha/line_ratio_plots' directory.")

# ============================================================================
# Compute full grid for MCMC inference and save to .dat files
# ============================================================================

print("\nComputing full 3D grid [r_ref, k, beta] for MCMC inference...")
print(f"k_grid: {n_k} values from {k_min} to {k_max}")
print(f"beta_grid: {len(beta_grid_mcmc)} values from {beta_grid_mcmc.min():.1f} to {beta_grid_mcmc.max():.1f}")
print(f"r_ref_grid: {len(r_ref_grid)} values")
print(f"Fixed gamma = {gamma_mcmc}")

# Directory for data files
data_dir = "data/alpha/mcmc_data"
os.makedirs(data_dir, exist_ok=True)

# Loop over ALL r_ref values, beta values, and k values
n_total = len(r_ref_grid) * len(beta_grid_mcmc)
n_done = 0

for r_ref_mcmc_val in r_ref_grid:
    log_r_ref_mcmc = np.log10(r_ref_mcmc_val)

    for beta in beta_grid_mcmc:
        n_done += 1
        print(f"  [{n_done}/{n_total}] r_ref=10^{log_r_ref_mcmc:.2f}, beta={beta:.1f}...")

        mcmc_results = {}
        for k in k_grid:
            Z_r = create_Z_profile(r, k, Z_min, Z_max)
            results = compute_line_ratios_vs_Q(r_ref_mcmc_val, Z_r, gamma_mcmc, breathing_factor=beta)
            mcmc_results[k] = results

        # Save to .dat file
        # Format: line_ratios_k_grid_gamma-1.2_rref{log_r_ref:.2f}_beta{beta:.2f}.dat
        output_file = os.path.join(data_dir,
            f"line_ratios_k_grid_gamma{gamma_mcmc:.1f}_rref{log_r_ref_mcmc:.2f}_beta{beta:.2f}.dat")

        with open(output_file, 'w') as f:
            # Header
            f.write(f"# Line ratios for 3D grid [r_ref, k, beta] MCMC inference\n")
            f.write(f"# gamma = {gamma_mcmc:.1f} (fixed), r_ref = {r_ref_mcmc_val:.2e} cm (log10 = {log_r_ref_mcmc:.2f})\n")
            f.write(f"# beta = {beta:.2f} (breathing factor)\n")
            f.write(f"# k_grid: {n_k} values from {k_min} to {k_max}\n")
            f.write(f"# Columns: k, logQ, MgII/CIV, (SiIV+OIV)/CIV\n")
            f.write(f"# Each block corresponds to one k value\n")
            f.write(f"# Format: k logQ ratio1 ratio2\n")
            f.write("#\n")

            for k in k_grid:
                f.write(f"\n# k = {k:.6f}\n")
                logQ = mcmc_results[k]['logQ']
                mg2c4 = mcmc_results[k]['MgII/CIV']
                si4o4c4 = mcmc_results[k]['(SiIV+OIV)/CIV']

                for q, r1, r2 in zip(logQ, mg2c4, si4o4c4):
                    f.write(f"{k:.6f} {q:.4f} {r1:.6e} {r2:.6e}\n")

        print(f"    Saved to '{output_file}'")

print(f"\nMCMC data saved for 3D grid [{len(r_ref_grid)} r_ref × {len(beta_grid_mcmc)} beta × {n_k} k] in '{data_dir}' directory")

# ============================================================================
# Create 2D interpolation plot (checking for correct axes)
# ============================================================================

print("\nCreating 2D interpolation plot...")

# Create fine grids for smooth 2D plot
phi_fine = np.linspace(min(phi_grid), max(phi_grid), 200)
Z_fine = np.linspace(min(Z_grid), max(Z_grid), 200)

# Create meshgrids for plotting
Phi_fine, Z_fine_mesh = np.meshgrid(phi_fine, Z_fine)

# Evaluate interpolation functions explicitly to ensure correct ordering
# Grid shape: (len(Z_fine), len(phi_fine))
# i indexes Z, j indexes phi
mg2c4_grid = np.zeros((len(Z_fine), len(phi_fine)))
si4o4c4_grid = np.zeros((len(Z_fine), len(phi_fine)))

for i, Z_val in enumerate(Z_fine):
    for j, phi_val in enumerate(phi_fine):
        # Evaluate interpolation at (phi, Z)
        # Note: grid=False returns scalar, not array
        Mg2_val = float(Mg2_interp_2D(phi_val, Z_val, grid=False))
        C4_val = float(C4_interp_2D(phi_val, Z_val, grid=False))
        Si4_val = float(Si4_interp_2D(phi_val, Z_val, grid=False))
        O4_val = float(O4_interp_2D(phi_val, Z_val, grid=False))
        
        mg2c4_grid[i, j] = Mg2_val / C4_val
        si4o4c4_grid[i, j] = (Si4_val + O4_val) / C4_val

# Verify: MgII/CIV should increase with Z (high Z should have higher ratios)
# Check a few points
Z_low_idx = 0
Z_high_idx = -1
phi_mid_idx = len(phi_fine) // 2
print(f"\nVerifying interpolation:")
print(f"At phi = {phi_fine[phi_mid_idx]:.2f}:")
print(f"  Z = {Z_fine[Z_low_idx]:.2f} -> MgII/CIV = {mg2c4_grid[Z_low_idx, phi_mid_idx]:.4f}")
print(f"  Z = {Z_fine[Z_high_idx]:.2f} -> MgII/CIV = {mg2c4_grid[Z_high_idx, phi_mid_idx]:.4f}")
print(f"Ratio (high Z / low Z) = {mg2c4_grid[Z_high_idx, phi_mid_idx] / mg2c4_grid[Z_low_idx, phi_mid_idx]:.2f}")

# Create 2D plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left panel: MgII/CIV
ax = axes[0]
im1 = ax.pcolormesh(Phi_fine, Z_fine_mesh, np.log10(mg2c4_grid),
                    shading='auto', cmap=cmap, vmin=np.log10(mg2c4_grid).min(),
                    vmax=np.log10(mg2c4_grid).max())
ax.set_xlabel(r'$\log \phi\ [\rm photons\ s^{-1}\ cm^{-2}]$', fontsize=16)
ax.set_ylabel(r'$Z\ [{\rm Z}_\odot]$', fontsize=16)
ax.set_title(r'$\log(\rm MgII / CIV)$', fontsize=16)
plt.colorbar(im1, ax=ax, label=r'$\log(\rm MgII / CIV)$')
ax.tick_params(top=True, right=True, axis='both', which='major', 
               length=8, width=1.5, direction='in')
ax.tick_params(top=True, right=True, axis='both', which='minor', 
               length=4, width=1, direction='in')
ax.minorticks_on()

# Right panel: (SiIV+OIV)/CIV
ax = axes[1]
im2 = ax.pcolormesh(Phi_fine, Z_fine_mesh, np.log10(si4o4c4_grid),
                    shading='auto', cmap=cmap, vmin=np.log10(si4o4c4_grid).min(),
                    vmax=np.log10(si4o4c4_grid).max())
ax.set_xlabel(r'$\log \phi\ [\rm photons\ s^{-1}\ cm^{-2}]$', fontsize=16)
ax.set_ylabel(r'$Z\ [{\rm Z}_\odot]$', fontsize=16)
ax.set_title(r'$\log[(\rm SiIV + OIV) / CIV]$', fontsize=16)
plt.colorbar(im2, ax=ax, label=r'$\log[(\rm SiIV + OIV) / CIV]$')
ax.tick_params(top=True, right=True, axis='both', which='major', 
               length=8, width=1.5, direction='in')
ax.tick_params(top=True, right=True, axis='both', which='minor', 
               length=4, width=1, direction='in')
ax.minorticks_on()

plt.tight_layout()
output_file_2d = "line_ratio_2d_interpolation.png"
plt.savefig(output_file_2d, dpi=300, bbox_inches='tight')
plt.close()

print(f"2D interpolation plot saved as '{output_file_2d}'")

# ============================================================================
# Simple diagnostic: Z(r) profiles colored by line ratio, with window markers
# Shows how the averaging window moves in r-space as Q changes
# ============================================================================

print("\nCreating simple Z(r) diagnostic plots with line-ratio coloring...")

log_Q_example = [54.0, 55.0, 56.0]
Q_example = 10**np.array(log_Q_example)
marker_styles_Q = ['*', 's', 'D']  # star, square, diamond for different Q
marker_sizes_Q = [200, 100, 100]

r_pc = r / cm_to_pc  # Convert to parsecs
log_r_pc = np.log10(r_pc)

# Pre-compute global colorbar range across all r_ref values
print("  Computing global colorbar range across all r_ref values...")
global_log_ratios_mg2c4 = []
global_log_ratios_si4o4c4 = []
for r_ref_val in r_ref_grid:
    phi_center_val = np.log10(Q_ref) - 2 * np.log10(r_ref_val)
    phi_center_clip = np.clip(phi_center_val, phi_grid.min(), phi_grid.max())
    for k in k_values_plot:
        Z_r = create_Z_profile(r, k, Z_min, Z_max)
        # MgII/CIV
        ratio_mg2c4 = np.array([
            float(Mg2_interp_2D(phi_center_clip, Z_r[i], grid=False)) /
            float(C4_interp_2D(phi_center_clip, Z_r[i], grid=False))
            for i in range(len(r))
        ])
        global_log_ratios_mg2c4.extend(np.log10(ratio_mg2c4))
        # (SiIV+OIV)/CIV
        ratio_si4o4c4 = np.array([
            (float(Si4_interp_2D(phi_center_clip, Z_r[i], grid=False)) +
             float(O4_interp_2D(phi_center_clip, Z_r[i], grid=False))) /
            float(C4_interp_2D(phi_center_clip, Z_r[i], grid=False))
            for i in range(len(r))
        ])
        global_log_ratios_si4o4c4.extend(np.log10(ratio_si4o4c4))

norm_mg2c4_global = plt.Normalize(min(global_log_ratios_mg2c4), max(global_log_ratios_mg2c4))
norm_si4o4c4_global = plt.Normalize(min(global_log_ratios_si4o4c4), max(global_log_ratios_si4o4c4))
print(f"  MgII/CIV: log ratio range = [{min(global_log_ratios_mg2c4):.2f}, {max(global_log_ratios_mg2c4):.2f}]")
print(f"  (SiIV+OIV)/CIV: log ratio range = [{min(global_log_ratios_si4o4c4):.2f}, {max(global_log_ratios_si4o4c4):.2f}]")

for r_ref_val in r_ref_grid:
    log_r_ref_val = np.log10(r_ref_val)
    phi_center_val = np.log10(Q_ref) - 2 * log_r_ref_val
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax_idx, (ax, ratio_name, ratio_label, norm) in enumerate([
        (axes[0], 'MgII/CIV', r'$\rm MgII/CIV$', norm_mg2c4_global),
        (axes[1], '(SiIV+OIV)/CIV', r'$\rm (SiIV+OIV)/CIV$', norm_si4o4c4_global)
    ]):
        # With breathing effect, φ_center is constant. Evaluate ratio at constant φ = φ_center
        # This shows the actual ratio being sampled at the window center
        phi_center_clip = np.clip(phi_center_val, phi_grid.min(), phi_grid.max())
        
        # Second pass: plot lines
        for ik, (k, label) in enumerate(zip(k_values_plot, labels_gradient)):
            Z_r = create_Z_profile(r, k, Z_min, Z_max)
            
            # Ratio at constant φ = φ_center, varying Z along the profile
            if ratio_name == 'MgII/CIV':
                ratio_vals = np.array([
                    float(Mg2_interp_2D(phi_center_clip, Z_r[i], grid=False)) /
                    float(C4_interp_2D(phi_center_clip, Z_r[i], grid=False))
                    for i in range(len(r))
                ])
            else:
                ratio_vals = np.array([
                    (float(Si4_interp_2D(phi_center_clip, Z_r[i], grid=False)) +
                     float(O4_interp_2D(phi_center_clip, Z_r[i], grid=False))) /
                    float(C4_interp_2D(phi_center_clip, Z_r[i], grid=False))
                    for i in range(len(r))
                ])
            
            log_ratio = np.log10(ratio_vals)
            points = np.array([log_r_pc, Z_r]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Thin line for full curve (outside ±1σ)
            lc_thin = LineCollection(segments, cmap=cmap, norm=norm, linewidth=1.5, alpha=0.5)
            lc_thin.set_array(log_ratio[:-1])
            ax.add_collection(lc_thin)
            
            # Thick lines for ±1σ regions for each Q
            for Q in Q_example:
                r_center = r_ref_val * (Q / Q_ref)**0.5
                r_low = r_center / 10**sigma
                r_high = r_center * 10**sigma
                i_low = np.argmin(np.abs(r - r_low))
                i_high = np.argmin(np.abs(r - r_high))
                idx_lo, idx_hi = min(i_low, i_high), max(i_low, i_high)
                
                if idx_hi > idx_lo:
                    seg_thick = segments[idx_lo:idx_hi]
                    lc_thick = LineCollection(seg_thick, cmap=cmap, norm=norm, linewidth=5.0, alpha=0.9)
                    lc_thick.set_array(log_ratio[idx_lo:idx_hi])
                    ax.add_collection(lc_thick)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(rf'$\log({ratio_label})$')
        
        # Plot window markers for each Q
        for iq, (Q, marker, ms) in enumerate(zip(Q_example, marker_styles_Q, marker_sizes_Q)):
            r_center = r_ref_val * (Q / Q_ref)**0.5
            r_low = r_center / 10**sigma
            r_high = r_center * 10**sigma
            
            for ik, k in enumerate(k_values_plot):
                Z_r = create_Z_profile(r, k, Z_min, Z_max)
                i_center = np.argmin(np.abs(r - r_center))
                i_low = np.argmin(np.abs(r - r_low))
                i_high = np.argmin(np.abs(r - r_high))
                
                ax.scatter(log_r_pc[i_center], Z_r[i_center], 
                          c='white', s=ms, marker=marker, edgecolors='k', linewidths=1.5, zorder=10)
                ax.scatter(log_r_pc[i_low], Z_r[i_low], 
                          c='white', s=50, marker='o', edgecolors='k', linewidths=1, zorder=9, alpha=0.8)
                ax.scatter(log_r_pc[i_high], Z_r[i_high], 
                          c='white', s=50, marker='o', edgecolors='k', linewidths=1, zorder=9, alpha=0.8)
        
        ax.set_xlim(log_r_pc.min(), log_r_pc.max())
        ax.set_ylim(Z_min - 0.5, Z_max + 0.5)
        ax.set_xlabel(r'$\log r\ \mathrm{[pc]}$', fontsize=16)
        ax.set_ylabel(r'$Z\ [{\rm Z}_\odot]$', fontsize=16)
        ax.set_title(rf'{ratio_label} | $\log r_{{\rm ref}}={log_r_ref_val:.2f}$, $\phi_{{\rm center}}={phi_center_val:.1f}$', fontsize=14)
        ax.tick_params(top=True, right=True, axis='both', which='major', 
                       length=8, width=1.5, direction='in')
        ax.tick_params(top=True, right=True, axis='both', which='minor', 
                       length=4, width=1, direction='in')
        ax.minorticks_on()
    
    # Legend for Q markers
    leg_handles_simple = [
        Line2D([0], [0], marker='*', color='w', markeredgecolor='k', markersize=15, linestyle='', label=r'$\log Q=54$'),
        Line2D([0], [0], marker='s', color='w', markeredgecolor='k', markersize=10, linestyle='', label=r'$\log Q=55$'),
        Line2D([0], [0], marker='D', color='w', markeredgecolor='k', markersize=10, linestyle='', label=r'$\log Q=56$'),
        Line2D([0], [0], marker='o', color='w', markeredgecolor='k', markersize=8, linestyle='', label=r'$\pm 1\sigma$ extent'),
    ]
    axes[1].legend(handles=leg_handles_simple, fontsize=10, loc='upper right')
    
    plt.tight_layout()
    output_simple = f"line_ratio_Zr_window_rref_log{log_r_ref_val:.2f}.png"
    plt.savefig(output_simple, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_simple}")

print("Simple Z(r) diagnostic plots complete.")

# ============================================================================
# Diagnostic: overlay (phi, Z) paths on 2D interpolation for example Q values
# Create separate plots for different r_ref values
# ============================================================================

print("\nCreating diagnostic overlay plots for different r_ref values...")
log_Q_example = [54.0, 55.0, 56.0]  # Example Q values spanning the plot range
Q_example = 10**np.array(log_Q_example)
linestyles_Q = ['-', '--', ':']  # Q=54 solid, Q=55 dashed, Q=56 dotted

# Use the r_ref values computed earlier (valid range for ±1σ within φ grid)
r_ref_diag_list = list(r_ref_grid)
log_r_ref_diag_list = [np.log10(rr) for rr in r_ref_diag_list]

for r_ref_diag, log_r_ref_diag in zip(r_ref_diag_list, log_r_ref_diag_list):
    # Compute phi at window center for this r_ref
    phi_center = np.log10(Q_ref) - 2 * log_r_ref_diag
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left panel: MgII/CIV with paths
    ax = axes[0]
    im1 = ax.pcolormesh(Phi_fine, Z_fine_mesh, np.log10(mg2c4_grid),
                        shading='auto', cmap=cmap, vmin=np.log10(mg2c4_grid).min(),
                        vmax=np.log10(mg2c4_grid).max())
    ax.set_xlabel(r'$\log \phi\ [\rm photons\ s^{-1}\ cm^{-2}]$', fontsize=16)
    ax.set_ylabel(r'$Z\ [{\rm Z}_\odot]$', fontsize=16)
    ax.set_title(rf'$\log(\rm MgII / CIV)$ | $\log r_{{\rm ref}}={log_r_ref_diag:.1f}$, $\phi_{{\rm center}}={phi_center:.1f}$', fontsize=14)
    plt.colorbar(im1, ax=ax, label=r'$\log(\rm MgII / CIV)$')
    
    # Marker styles consistent with Z(r) plot: star, square, diamond for Q=54, 55, 56
    # Smaller sizes for 2D plots to avoid clutter
    marker_styles_Q_diag = ['*', 's', 'D']
    marker_sizes_Q_diag = [100, 60, 60]
    
    for ik, (k, color, label) in enumerate(zip(k_values_plot, colors_gradient, labels_gradient)):
        Z_r = create_Z_profile(r, k, Z_min, Z_max)
        for iq, (Q, ls, mkr, mks) in enumerate(zip(Q_example, linestyles_Q, marker_styles_Q_diag, marker_sizes_Q_diag)):
            phi_r = np.log10(Q) - 2 * np.log10(r)
            # Mark weighted center: r_center = r_ref * (Q/Q_ref)^0.5
            r_center = r_ref_diag * (Q / Q_ref)**0.5
            i_center = np.argmin(np.abs(r - r_center))
            # Mark ±1σ extent of Gaussian window (sigma is in log r)
            r_low = r_center / 10**sigma   # r at -1σ
            r_high = r_center * 10**sigma  # r at +1σ
            i_low = np.argmin(np.abs(r - r_low))
            i_high = np.argmin(np.abs(r - r_high))
            # Plot thin line outside ±1σ, thick line inside ±1σ
            ax.plot(phi_r, Z_r, color=color, linestyle=ls, linewidth=1.0, alpha=0.5)
            # Thick line for ±1σ region (ensure i_low < i_high)
            idx_lo, idx_hi = min(i_low, i_high), max(i_low, i_high)
            ax.plot(phi_r[idx_lo:idx_hi+1], Z_r[idx_lo:idx_hi+1], 
                    color=color, linestyle=ls, linewidth=3.5, alpha=0.9)
            # Markers - use consistent style for each Q
            ax.scatter(phi_r[i_center], Z_r[i_center], c='white', s=mks, marker=mkr,
                       edgecolors='k', linewidths=1, zorder=5)
            ax.scatter(phi_r[i_low], Z_r[i_low], c='white', s=30, marker='o',
                       edgecolors='k', linewidths=0.5, zorder=4, alpha=0.8)
            ax.scatter(phi_r[i_high], Z_r[i_high], c='white', s=30, marker='o',
                       edgecolors='k', linewidths=0.5, zorder=4, alpha=0.8)
    
    # Legend
    leg_handles = [Line2D([0], [0], color=c, lw=2.5, label=l) for c, l in zip(colors_gradient, labels_gradient)]
    leg_handles.append(Line2D([0], [0], marker='*', color='w', markeredgecolor='k', markersize=10, linestyle='', label=r'$\log Q=54$'))
    leg_handles.append(Line2D([0], [0], marker='s', color='w', markeredgecolor='k', markersize=7, linestyle='', label=r'$\log Q=55$'))
    leg_handles.append(Line2D([0], [0], marker='D', color='w', markeredgecolor='k', markersize=7, linestyle='', label=r'$\log Q=56$'))
    leg_handles.append(Line2D([0], [0], marker='o', color='w', markeredgecolor='k', markersize=5, linestyle='', label=r'$\pm 1\sigma$ extent'))
    ax.legend(handles=leg_handles, fontsize=8, loc='upper right')
    ax.set_xlim(phi_fine.min(), phi_fine.max())
    ax.set_ylim(Z_fine.min(), Z_fine.max())
    ax.tick_params(top=True, right=True, axis='both', which='major', 
                   length=8, width=1.5, direction='in')
    ax.tick_params(top=True, right=True, axis='both', which='minor', 
                   length=4, width=1, direction='in')
    ax.minorticks_on()
    
    # Right panel: (SiIV+OIV)/CIV with paths
    ax = axes[1]
    im2 = ax.pcolormesh(Phi_fine, Z_fine_mesh, np.log10(si4o4c4_grid),
                        shading='auto', cmap=cmap, vmin=np.log10(si4o4c4_grid).min(),
                        vmax=np.log10(si4o4c4_grid).max())
    ax.set_xlabel(r'$\log \phi\ [\rm photons\ s^{-1}\ cm^{-2}]$', fontsize=16)
    ax.set_ylabel(r'$Z\ [{\rm Z}_\odot]$', fontsize=16)
    ax.set_title(rf'$\log[(\rm SiIV + OIV) / CIV]$ | $\log r_{{\rm ref}}={log_r_ref_diag:.1f}$', fontsize=14)
    plt.colorbar(im2, ax=ax, label=r'$\log[(\rm SiIV + OIV) / CIV]$')
    
    for ik, (k, color, label) in enumerate(zip(k_values_plot, colors_gradient, labels_gradient)):
        Z_r = create_Z_profile(r, k, Z_min, Z_max)
        for iq, (Q, ls, mkr, mks) in enumerate(zip(Q_example, linestyles_Q, marker_styles_Q_diag, marker_sizes_Q_diag)):
            phi_r = np.log10(Q) - 2 * np.log10(r)
            r_center = r_ref_diag * (Q / Q_ref)**0.5
            i_center = np.argmin(np.abs(r - r_center))
            # Mark ±1σ extent of Gaussian window
            r_low = r_center / 10**sigma
            r_high = r_center * 10**sigma
            i_low = np.argmin(np.abs(r - r_low))
            i_high = np.argmin(np.abs(r - r_high))
            # Plot thin line outside ±1σ, thick line inside ±1σ
            ax.plot(phi_r, Z_r, color=color, linestyle=ls, linewidth=1.0, alpha=0.5)
            idx_lo, idx_hi = min(i_low, i_high), max(i_low, i_high)
            ax.plot(phi_r[idx_lo:idx_hi+1], Z_r[idx_lo:idx_hi+1], 
                    color=color, linestyle=ls, linewidth=3.5, alpha=0.9)
            # Markers - use consistent style for each Q (smaller sizes for 2D plot)
            ax.scatter(phi_r[i_center], Z_r[i_center], c='white', s=mks, marker=mkr,
                       edgecolors='k', linewidths=1, zorder=5)
            ax.scatter(phi_r[i_low], Z_r[i_low], c='white', s=30, marker='o',
                       edgecolors='k', linewidths=0.5, zorder=4, alpha=0.8)
            ax.scatter(phi_r[i_high], Z_r[i_high], c='white', s=30, marker='o',
                       edgecolors='k', linewidths=0.5, zorder=4, alpha=0.8)
    
    ax.legend(handles=leg_handles, fontsize=8, loc='upper right')
    ax.set_xlim(phi_fine.min(), phi_fine.max())
    ax.set_ylim(Z_fine.min(), Z_fine.max())
    ax.tick_params(top=True, right=True, axis='both', which='major', 
                   length=8, width=1.5, direction='in')
    ax.tick_params(top=True, right=True, axis='both', which='minor', 
                   length=4, width=1, direction='in')
    ax.minorticks_on()
    
    plt.tight_layout()
    output_file_diag = f"line_ratio_2d_paths_rref_log{log_r_ref_diag:.2f}.png"
    plt.savefig(output_file_diag, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file_diag}")

print("Diagnostic overlay plots complete.")

# ============================================================================
# Create line ratio vs Q plots for multiple r_ref values and breathing factors
# ============================================================================

print("\nCreating line ratio vs Q plots for different r_ref and breathing values...")
gamma1, gamma2 = gammas_plot

# Use the same r_ref values as the diagnostic plots
for r_ref_summary in r_ref_diag_list:
    log_rref = np.log10(r_ref_summary)
    for beta in breathing_factors_plot:
        # phi_center depends on breathing: for β=1 it's constant, for β<1 it varies with Q
        phi_center_ref = np.log10(Q_ref) - 2 * log_rref  # φ at Q_ref
        results_gamma = all_results[r_ref_summary][beta]
        beta_str = f"beta{beta:.1f}"
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(rf'{breathing_labels[beta]} ($\beta={beta}$) | $\log r_{{\rm ref}} = {log_rref:.1f}$', 
                     fontsize=16, y=1.02)
        
        # Left panel: MgII/CIV
        ax = axes[0]
        for label in results_gamma[gamma1]:
            color = results_gamma[gamma1][label]['color']
            logQ = results_gamma[gamma1][label]['logQ']
            ratio_1 = results_gamma[gamma1][label]['MgII/CIV']
            ratio_2 = results_gamma[gamma2][label]['MgII/CIV']
            
            ax.fill_between(logQ, np.minimum(ratio_1, ratio_2),
                           np.maximum(ratio_1, ratio_2),
                           color=color, alpha=0.6, label=label)
        
        ax.set_xlabel(r'$\log Q$')
        ax.set_ylabel(r'$\rm MgII / CIV$')
        ax.set_yscale('log')
        ax.minorticks_on()
        ax.tick_params(top=True, right=True, axis='both', which='major', 
                       length=8, width=1.5, direction='in')
        ax.tick_params(top=True, right=True, axis='both', which='minor', 
                       length=4, width=1, direction='in')
        ax.legend(title=r"$\rm Z\ Profile$", fontsize=10)
        
        # Right panel: (SiIV+OIV)/CIV
        ax = axes[1]
        for label in results_gamma[gamma1]:
            color = results_gamma[gamma1][label]['color']
            logQ = results_gamma[gamma1][label]['logQ']
            ratio_1 = results_gamma[gamma1][label]['(SiIV+OIV)/CIV']
            ratio_2 = results_gamma[gamma2][label]['(SiIV+OIV)/CIV']
            
            ax.fill_between(logQ, np.minimum(ratio_1, ratio_2),
                           np.maximum(ratio_1, ratio_2),
                           color=color, alpha=0.6, label=label)
        
        ax.set_xlabel(r'$\log Q$')
        ax.set_ylabel(r'$\rm (SiIV + OIV) / CIV$')
        ax.minorticks_on()
        ax.tick_params(top=True, right=True, axis='both', which='major', 
                       length=8, width=1.5, direction='in')
        ax.tick_params(top=True, right=True, axis='both', which='minor', 
                       length=4, width=1, direction='in')
        ax.legend(title=r"$\rm Z\ Profile$", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'line_ratios_summary_rref_log{log_rref:.2f}_{beta_str}.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: line_ratios_summary_rref_log{log_rref:.2f}_{beta_str}.png")

print("Line ratio vs Q plots complete.")
