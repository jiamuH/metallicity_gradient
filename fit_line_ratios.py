#!/usr/bin/env python3
"""
Fit observed line ratios using pre-computed Cloudy models.

This script:
1. Loads observed line ratios for rm002 (from observed_line_ratio_data/)
2. Loads pre-computed model grid from mcmc_data/
3. Uses NGP (Nearest Grid Point) interpolation for MCMC
4. Fits for k (metallicity gradient) and conversion factor from Q to F1350
5. Fits mg2c4 and (si4+o4)/c4 ratios separately first, then jointly

Requirements:
- emcee: pip install emcee
- corner: pip install corner
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize
from scipy.interpolate import interp1d, griddata
import emcee
import corner
import os
import re
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy import constants as const
from astropy.stats import sigma_clip

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
rm_id = "rm266"
model_file_pattern = "mcmc_data/line_ratios_k_grid_gamma-1.2_rref{rref:.2f}_beta{beta:.2f}.dat"
gamma_fixed = -1.2  # Fixed gamma (Korista & Goad 2019)
beta_values = np.arange(0.0, 1.05, 0.1)  # Breathing factor grid: 0.0 to 1.0
# rref_values will be populated from available files (see load_model_grid)
observed_data_file = f"observed_line_ratio_data/{rm_id}_line_ratios.dat"
output_dir = "mcmc_fits"
plot_dir = "nagao_ratio_plots"

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Model uncertainty (fractional error from grid spacing/interpolation)
MODEL_UNCERTAINTY = 0.1  # 10% uncertainty

# Cosmology (using standard values)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Wavelength of 1350A in rest frame
LAMBDA_1350 = 1350.0 * u.Angstrom  # Rest frame wavelength


def load_observed_data(filename):
    """
    Load observed line ratio data and extract redshift.
    
    Returns:
    --------
    data : dict
        Dictionary with keys: 'mg2_c4', 'si4_c4', 'c3_c4', 'redshift'
        Each ratio contains: 'f1350', 'ratio', 'ratio_err', 'mjd'
    """
    data = {'mg2_c4': None, 'si4_c4': None, 'c3_c4': None, 'redshift': None}
    redshift = None
    
    with open(filename, 'r') as f:
        for line in f:
            # Extract redshift from header (format: # Line ratios for rm002 (z = 1.755350))
            if line.startswith('#') and ('z =' in line or '(z =' in line):
                match = re.search(r'\(?\s*z\s*=\s*([\d.]+)\)?', line)
                if match:
                    redshift = float(match.group(1))
                    data['redshift'] = redshift
                    continue
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            f1350 = float(parts[0])
            ratio = float(parts[1])
            ratio_err = float(parts[2])
            mjd = float(parts[3])
            ratio_type = parts[4]
            
            if ratio_type == 'mg2_c4':
                if data['mg2_c4'] is None:
                    data['mg2_c4'] = {'f1350': [], 'ratio': [], 'ratio_err': [], 'mjd': []}
                data['mg2_c4']['f1350'].append(f1350)
                data['mg2_c4']['ratio'].append(ratio)
                data['mg2_c4']['ratio_err'].append(ratio_err)
                data['mg2_c4']['mjd'].append(mjd)
            
            elif ratio_type == 'si4_c4':
                if data['si4_c4'] is None:
                    data['si4_c4'] = {'f1350': [], 'ratio': [], 'ratio_err': [], 'mjd': []}
                data['si4_c4']['f1350'].append(f1350)
                data['si4_c4']['ratio'].append(ratio)
                data['si4_c4']['ratio_err'].append(ratio_err)
                data['si4_c4']['mjd'].append(mjd)
            
            elif ratio_type == 'c3_c4':
                if data['c3_c4'] is None:
                    data['c3_c4'] = {'f1350': [], 'ratio': [], 'ratio_err': [], 'mjd': []}
                data['c3_c4']['f1350'].append(f1350)
                data['c3_c4']['ratio'].append(ratio)
                data['c3_c4']['ratio_err'].append(ratio_err)
                data['c3_c4']['mjd'].append(mjd)
    
    # Convert to numpy arrays
    for key in data:
        if data[key] is not None and key != 'redshift':
            data[key] = {k: np.array(v) for k, v in data[key].items()}

    # Remove outlier data points with very low flux (< 1e-19)
    flux_floor = 1e-19
    for key in ['mg2_c4', 'si4_c4', 'c3_c4']:
        if data[key] is not None:
            mask = data[key]['f1350'] >= flux_floor
            n_removed = np.sum(~mask)
            if n_removed > 0:
                print(f"  Removed {n_removed} points with F1350 < {flux_floor:.0e} from {key}")
                data[key] = {k: v[mask] for k, v in data[key].items()}
                if len(data[key]['f1350']) == 0:
                    data[key] = None

    # Sigma-clip outliers in ratio values AND f1350 values
    for key in ['mg2_c4', 'si4_c4', 'c3_c4']:
        if data[key] is not None and len(data[key]['ratio']) > 5:
            # Clip on ratios
            clipped_ratio = sigma_clip(data[key]['ratio'], sigma=3, maxiters=3)
            mask = ~clipped_ratio.mask
            n_removed_ratio = np.sum(~mask)
            if n_removed_ratio > 0:
                print(f"  Sigma-clipped {n_removed_ratio} outlier(s) from {key} ratios")
                data[key] = {k: v[mask] for k, v in data[key].items()}
            # Clip on f1350 (removes bad photometry epochs)
            if data[key] is not None and len(data[key]['f1350']) > 5:
                clipped_f1350 = sigma_clip(data[key]['f1350'], sigma=3, maxiters=3)
                mask_f = ~clipped_f1350.mask
                n_removed_f1350 = np.sum(~mask_f)
                if n_removed_f1350 > 0:
                    print(f"  Sigma-clipped {n_removed_f1350} outlier(s) from {key} f1350")
                    data[key] = {k: v[mask_f] for k, v in data[key].items()}
                if len(data[key]['f1350']) == 0:
                    data[key] = None

    return data


def compute_photon_flux_conversion(redshift):
    """
    Compute conversion factor from Q (ionizing photon rate) to photon flux.
    
    photon_flux = Q / (4π * d_L^2)  [photons/s/cm^2]
    
    Returns log10(4π * d_L^2) for use in converting F1350 to Q.
    
    Parameters:
    -----------
    redshift : float
        Redshift of the object
    
    Returns:
    --------
    log_4pi_dL2 : float
        log10(4π * d_L^2) in cm^2
    """
    if redshift is None or redshift <= 0:
        # Return a default if redshift not available (won't be used)
        return None
    
    # Compute luminosity distance
    d_L = cosmo.luminosity_distance(redshift)
    d_L_cm = d_L.to(u.cm).value
    
    # photon_flux = Q / (4π * d_L^2)
    # So: Q = photon_flux * (4π * d_L^2)
    log_4pi_dL2 = np.log10(4 * np.pi * d_L_cm**2)
    
    return log_4pi_dL2


def load_model_grid(beta_values, rref_values, model_file_pattern):
    """
    Load pre-computed model grid for multiple (r_ref, beta) values.

    Parameters:
    -----------
    beta_values : array
        Array of beta (breathing factor) values to load
    rref_values : array
        Array of log10(r_ref) values to load
    model_file_pattern : str
        Pattern for model file names with {rref} and {beta} placeholders

    Returns:
    --------
    models : dict
        Dictionary with structure: models[rref][beta][k][logQ] = {'mg2c4': value, 'si4o4c4': value}
    k_values : array
        Array of k values
    logQ_values : array
        Array of logQ values
    rref_values_loaded : array
        Array of log10(r_ref) values that were successfully loaded
    beta_values_loaded : array
        Array of beta values that were successfully loaded
    """
    models = {}
    k_values_set = set()
    logQ_values_set = set()
    rref_values_loaded = set()
    beta_values_loaded = set()

    for rref in rref_values:
        for beta in beta_values:
            filename = model_file_pattern.format(rref=rref, beta=beta)
            if not os.path.exists(filename):
                print(f"Warning: Model file not found: {filename}, skipping rref={rref:.2f}, beta={beta:.2f}")
                continue

            rref_values_loaded.add(rref)
            beta_values_loaded.add(beta)

            if rref not in models:
                models[rref] = {}
            if beta not in models[rref]:
                models[rref][beta] = {}

            current_k = None

            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('# k ='):
                        current_k = float(line.split('=')[1].strip())
                        k_values_set.add(current_k)
                        if current_k not in models[rref][beta]:
                            models[rref][beta][current_k] = {}

                    elif not line.startswith('#') and line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            k = float(parts[0])
                            logQ = float(parts[1])
                            mg2c4 = float(parts[2])
                            si4o4c4 = float(parts[3])

                            logQ_values_set.add(logQ)
                            models[rref][beta][k][logQ] = {'mg2c4': mg2c4, 'si4o4c4': si4o4c4}

    k_values = np.array(sorted(k_values_set))
    logQ_values = np.array(sorted(logQ_values_set))
    rref_values_loaded = np.array(sorted(rref_values_loaded))
    beta_values_loaded = np.array(sorted(beta_values_loaded))

    return models, k_values, logQ_values, rref_values_loaded, beta_values_loaded


def interpolate_model_grid(models, k_values, logQ_values, rref_values, beta_values,
                           n_k_interp=100, n_beta_interp=50):
    """
    Interpolate the k and beta grids to have more points for smoother MCMC sampling.

    For each r_ref, interpolate across (k, beta) at each logQ.

    Parameters:
    -----------
    models : dict
        Original model grid with structure: models[rref][beta][k][logQ]
    k_values : array
        Original k grid values
    logQ_values : array
        logQ grid values (unchanged)
    rref_values : array
        log10(r_ref) values (unchanged)
    beta_values : array
        Original beta grid values
    n_k_interp : int
        Number of k values in interpolated grid
    n_beta_interp : int
        Number of beta values in interpolated grid

    Returns:
    --------
    models_interp : dict
        Interpolated model grid with structure: models_interp[rref][beta][k][logQ]
    k_values_interp : array
        Interpolated k grid values
    beta_values_interp : array
        Interpolated beta grid values
    """
    k_min = k_values.min()
    k_max = k_values.max()
    k_values_interp = np.linspace(k_min, k_max, n_k_interp)

    beta_min = beta_values.min()
    beta_max = beta_values.max()
    beta_values_interp = np.linspace(beta_min, beta_max, n_beta_interp)

    models_interp = {}

    for rref in rref_values:
        if rref not in models:
            continue
        models_interp[rref] = {}

        # For each logQ value, interpolate across k and beta
        for logQ in logQ_values:
            points_k = []
            points_beta = []
            mg2c4_vals = []
            si4o4c4_vals = []

            for beta in beta_values:
                if beta not in models[rref]:
                    continue
                for k in k_values:
                    if k in models[rref][beta] and logQ in models[rref][beta][k]:
                        points_k.append(k)
                        points_beta.append(beta)
                        mg2c4_vals.append(models[rref][beta][k][logQ]['mg2c4'])
                        si4o4c4_vals.append(models[rref][beta][k][logQ]['si4o4c4'])

            if len(points_k) < 3:
                continue

            points = np.column_stack([points_k, points_beta])
            mg2c4_vals = np.array(mg2c4_vals)
            si4o4c4_vals = np.array(si4o4c4_vals)

            # Interpolate in log space for ratios
            log_mg2c4_vals = np.log10(mg2c4_vals)
            log_si4o4c4_vals = np.log10(si4o4c4_vals)

            # Create grid for interpolation
            k_grid_i, beta_grid_i = np.meshgrid(k_values_interp, beta_values_interp, indexing='ij')
            grid_points = np.column_stack([k_grid_i.ravel(), beta_grid_i.ravel()])

            # Interpolate using griddata
            log_mg2c4_interp = griddata(points, log_mg2c4_vals, grid_points, method='linear', fill_value=np.nan)
            log_si4o4c4_interp = griddata(points, log_si4o4c4_vals, grid_points, method='linear', fill_value=np.nan)

            # Reshape and store
            log_mg2c4_interp = log_mg2c4_interp.reshape(len(k_values_interp), len(beta_values_interp))
            log_si4o4c4_interp = log_si4o4c4_interp.reshape(len(k_values_interp), len(beta_values_interp))

            for i, k_interp in enumerate(k_values_interp):
                for j, beta_interp in enumerate(beta_values_interp):
                    if beta_interp not in models_interp[rref]:
                        models_interp[rref][beta_interp] = {}
                    if k_interp not in models_interp[rref][beta_interp]:
                        models_interp[rref][beta_interp][k_interp] = {}

                    if not np.isnan(log_mg2c4_interp[i, j]):
                        models_interp[rref][beta_interp][k_interp][logQ] = {
                            'mg2c4': 10**log_mg2c4_interp[i, j],
                            'si4o4c4': 10**log_si4o4c4_interp[i, j]
                        }

    return models_interp, k_values_interp, beta_values_interp


def ngp_interpolate(models, k_values, logQ_values, rref_values, beta_values, k, logQ, rref, beta):
    """
    Nearest Grid Point interpolation with uncertainty estimate.

    Parameters:
    -----------
    models : dict
        Model grid with structure: models[rref][beta][k][logQ]
    k_values : array
        Array of k values
    logQ_values : array
        Array of logQ values
    rref_values : array
        Array of log10(r_ref) values
    beta_values : array
        Array of beta values
    k : float
        k value to interpolate
    logQ : float
        logQ value to interpolate
    rref : float
        log10(r_ref) value to interpolate
    beta : float
        beta value to interpolate

    Returns:
    --------
    result : dict or None
        Dictionary with 'mg2c4', 'si4o4c4', and their uncertainties, or None if out of bounds
    """
    # Find nearest r_ref
    rref_idx = np.argmin(np.abs(rref_values - rref))
    rref_nearest = rref_values[rref_idx]

    # Find nearest beta
    beta_idx = np.argmin(np.abs(beta_values - beta))
    beta_nearest = beta_values[beta_idx]

    # Find nearest k
    k_idx = np.argmin(np.abs(k_values - k))
    k_nearest = k_values[k_idx]

    # Find nearest logQ
    logQ_idx = np.argmin(np.abs(logQ_values - logQ))
    logQ_nearest = logQ_values[logQ_idx]

    # Check bounds
    if (rref_nearest not in models or
        beta_nearest not in models[rref_nearest] or
        k_nearest not in models[rref_nearest][beta_nearest] or
        logQ_nearest not in models[rref_nearest][beta_nearest][k_nearest]):
        return None

    model_ratio = models[rref_nearest][beta_nearest][k_nearest][logQ_nearest].copy()

    # Add uncertainty (fractional error)
    model_ratio['mg2c4_err'] = model_ratio['mg2c4'] * MODEL_UNCERTAINTY
    model_ratio['si4o4c4_err'] = model_ratio['si4o4c4'] * MODEL_UNCERTAINTY

    return model_ratio


def log_likelihood_single_ratio(params, observed_data, ratio_type, models, k_values, logQ_values,
                                  rref_values, beta_values, rref, log_4pi_dL2=None):
    """
    Compute log-likelihood for a single ratio type.

    Parameters:
    -----------
    params : array
        [k, beta, log_C_Q, offset]
        k: metallicity gradient
        beta: breathing factor
        log_C_Q: log10 of C_Q, where F1350 = C_Q * photon_flux
        offset: multiplicative offset between observed and model ratios
    observed_data : dict
        Observed data for the specific ratio type
    ratio_type : str
        'mg2_c4' or 'si4_c4'
    models : dict
        Model grid with structure: models[rref][beta][k][logQ]
    k_values : array
        k grid values
    logQ_values : array
        logQ grid values
    rref_values : array
        log10(r_ref) grid values
    beta_values : array
        beta grid values
    rref : float
        Fixed log10(r_ref) for this evaluation (nearest grid point used)
    log_4pi_dL2 : float
        log10(4π * d_L^2) from luminosity distance (if redshift available)

    Returns:
    --------
    log_likelihood : float
        Log-likelihood
    """
    k, beta, log_C_Q, offset = params

    # Check bounds
    if k < k_values.min() or k > k_values.max():
        return -np.inf
    if beta < beta_values.min() or beta > beta_values.max():
        return -np.inf

    if observed_data is None or len(observed_data['f1350']) == 0:
        return -np.inf

    f1350_obs = observed_data['f1350']
    ratio_obs = observed_data['ratio']
    ratio_err_obs = observed_data['ratio_err']

    # Convert F1350 to logQ
    if log_4pi_dL2 is not None:
        logQ_obs = np.log10(f1350_obs) + log_4pi_dL2 - log_C_Q
    else:
        logQ_obs = np.log10(f1350_obs) - log_C_Q

    log_likelihood = 0.0

    model_key = 'mg2c4' if ratio_type == 'mg2_c4' else 'si4o4c4'
    model_err_key = 'mg2c4_err' if ratio_type == 'mg2_c4' else 'si4o4c4_err'

    # Check if converted logQ values are within model grid bounds
    if logQ_obs.min() < logQ_values.min() - 0.1 or logQ_obs.max() > logQ_values.max() + 0.1:
        return -1e6

    # Compute density-based weights so dense F1350 regions don't dominate
    # Bin data in logQ and weight each point by 1/n_in_bin
    n_bins = max(5, len(logQ_obs) // 10)
    bin_edges = np.linspace(logQ_obs.min() - 1e-6, logQ_obs.max() + 1e-6, n_bins + 1)
    bin_indices = np.digitize(logQ_obs, bin_edges) - 1
    bin_counts = np.bincount(bin_indices, minlength=n_bins).astype(float)
    bin_counts[bin_counts == 0] = 1
    weights = 1.0 / np.sqrt(bin_counts[bin_indices])
    weights /= weights.sum()  # normalize so total weight = 1
    weights *= len(logQ_obs)  # scale so effective N is preserved

    for i, (f1350, ratio, err, logQ) in enumerate(zip(f1350_obs, ratio_obs, ratio_err_obs, logQ_obs)):
        model_result = ngp_interpolate(models, k_values, logQ_values, rref_values, beta_values,
                                       k, logQ, rref, beta)
        if model_result is None:
            return -1e6

        model_ratio = model_result[model_key]
        model_ratio_err = model_result[model_err_key]

        ratio_corrected = ratio / offset

        total_err = np.sqrt(err**2 + model_ratio_err**2)

        chi2 = ((ratio_corrected - model_ratio) / total_err) ** 2
        log_likelihood += -0.5 * chi2 * weights[i]

    return log_likelihood


def log_likelihood_joint(params, observed_data, models, k_values, logQ_values,
                         rref_values, beta_values, rref, log_4pi_dL2=None):
    """
    Compute joint log-likelihood for mg2c4 and (si4+o4)/c4 ratios.

    Parameters:
    -----------
    params : array
        [k, beta, log_C_Q, offset_mg2, offset_si4]
        k: metallicity gradient
        beta: breathing factor
        log_C_Q: log10 of C_Q, where F1350 = C_Q * photon_flux
        Separate offsets for each ratio type
    rref : float
        Fixed log10(r_ref) for this evaluation
    log_4pi_dL2 : float
        log10(4π * d_L^2) from luminosity distance (if redshift available)
    """
    k, beta, log_C_Q, offset_mg2, offset_si4 = params

    # Check bounds
    if k < k_values.min() or k > k_values.max():
        return -np.inf
    if beta < beta_values.min() or beta > beta_values.max():
        return -np.inf

    log_likelihood = 0.0

    # Process mg2_c4 data
    if observed_data['mg2_c4'] is not None:
        params_mg2 = [k, beta, log_C_Q, offset_mg2]
        ll_mg2 = log_likelihood_single_ratio(
            params_mg2, observed_data['mg2_c4'], 'mg2_c4', models, k_values, logQ_values,
            rref_values, beta_values, rref, log_4pi_dL2
        )
        if not np.isfinite(ll_mg2) or ll_mg2 < -1e5:
            return -np.inf
        log_likelihood += ll_mg2

    # Process si4_c4 data
    if observed_data['si4_c4'] is not None:
        params_si4 = [k, beta, log_C_Q, offset_si4]
        ll_si4 = log_likelihood_single_ratio(
            params_si4, observed_data['si4_c4'], 'si4_c4', models, k_values, logQ_values,
            rref_values, beta_values, rref, log_4pi_dL2
        )
        if not np.isfinite(ll_si4) or ll_si4 < -1e5:
            return -np.inf
        log_likelihood += ll_si4

    return log_likelihood


def log_prior(params, k_values, beta_values, logQ_values, log_4pi_dL2=None,
              obs_f1350_min=None, obs_f1350_max=None, is_joint=False):
    """
    Prior distribution for parameters.

    Parameters:
    -----------
    params : array
        For single ratio: [k, beta, log_C_Q, offset]
        For joint: [k, beta, log_C_Q, offset_mg2, offset_si4]
        k: metallicity gradient
        beta: breathing factor
        log_C_Q: log10 of C_Q, where F1350 = C_Q * photon_flux
    k_values : array
        k grid values (for bounds)
    beta_values : array
        beta grid values (for bounds)
    logQ_values : array
        logQ grid values (for computing C_Q prior)
    log_4pi_dL2 : float
        log10(4π * d_L^2) from luminosity distance
    obs_f1350_min : float
        Minimum observed F1350 value (for computing C_Q prior)
    obs_f1350_max : float
        Maximum observed F1350 value (for computing C_Q prior)
    is_joint : bool
        Whether this is a joint fit (with separate offsets)

    Returns:
    --------
    log_prior : float
        Log prior probability
    """
    k = params[0]
    beta = params[1]
    log_C_Q = params[2]

    # Gaussian prior on k centered at 0 (physical: efficient mixing → modest gradients)
    # Hard boundary at grid edges, Gaussian penalty within
    k_prior_sigma = 0.1
    if k < k_values.min() or k > k_values.max():
        return -np.inf

    # Uniform prior on beta (capped at 0.5 — breathing is not fully efficient)
    beta_max_prior = 0.5
    if beta < beta_values.min() or beta > beta_max_prior:
        return -np.inf

    # Prior on log_C_Q
    # Physical constraint: max observed flux should not convert to logQ > 56
    # (Q_ref = 1e56 is already a very luminous AGN)
    logQ_physical_max = 56.0
    if log_4pi_dL2 is not None and obs_f1350_min is not None and obs_f1350_max is not None:
        log_F1350_min = np.log10(obs_f1350_min)
        log_F1350_max = np.log10(obs_f1350_max)
        logQ_min = logQ_values.min()

        log_C_Q_min = log_F1350_max - logQ_physical_max + log_4pi_dL2 - 1.0  # 1 dex padding
        log_C_Q_max = log_F1350_min - logQ_min + log_4pi_dL2 + 1.0  # 1 dex padding

        if log_C_Q < log_C_Q_min or log_C_Q > log_C_Q_max:
            return -np.inf
    else:
        if log_C_Q < -16.0 or log_C_Q > -12.0:
            return -np.inf

    # Gaussian prior contribution from k
    log_prior_conv = -0.5 * (k / k_prior_sigma)**2

    # Prior on offset
    if is_joint:
        offset_mg2 = params[3]
        offset_si4 = params[4]
        if offset_mg2 <= 0 or offset_mg2 > 20 or offset_si4 <= 0 or offset_si4 > 20:
            return -np.inf
    else:
        offset = params[3]
        if offset <= 0 or offset > 20:
            return -np.inf

    return log_prior_conv


def log_posterior_single(params, observed_data, ratio_type, models, k_values, logQ_values,
                         rref_values, beta_values, rref, log_4pi_dL2=None):
    """Log posterior for single ratio."""
    obs_f1350_min = None
    obs_f1350_max = None
    if observed_data is not None and 'f1350' in observed_data and len(observed_data['f1350']) > 0:
        obs_f1350_min = observed_data['f1350'].min()
        obs_f1350_max = observed_data['f1350'].max()

    lp = log_prior(params, k_values, beta_values, logQ_values, log_4pi_dL2,
                   obs_f1350_min, obs_f1350_max, is_joint=False)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_single_ratio(params, observed_data, ratio_type, models, k_values,
                                            logQ_values, rref_values, beta_values, rref, log_4pi_dL2)


def log_posterior_joint(params, observed_data, models, k_values, logQ_values,
                        rref_values, beta_values, rref, log_4pi_dL2=None):
    """Log posterior for joint fit."""
    obs_f1350_min = None
    obs_f1350_max = None
    obs_f1350_all = []
    for ratio_type in ['mg2_c4', 'si4_c4']:
        if observed_data.get(ratio_type) is not None and 'f1350' in observed_data[ratio_type]:
            obs_f1350_all.extend(observed_data[ratio_type]['f1350'])
    if len(obs_f1350_all) > 0:
        obs_f1350_all = np.array(obs_f1350_all)
        obs_f1350_min = obs_f1350_all.min()
        obs_f1350_max = obs_f1350_all.max()

    lp = log_prior(params, k_values, beta_values, logQ_values, log_4pi_dL2,
                   obs_f1350_min, obs_f1350_max, is_joint=True)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_joint(params, observed_data, models, k_values, logQ_values,
                                     rref_values, beta_values, rref, log_4pi_dL2)


def run_mcmc(observed_data, ratio_type, models, k_values, logQ_values,
              rref_values, beta_values, rref,
              k_init, beta_init, log_C_Q_init, offset_init=1.0, offset_init2=1.0,
              log_4pi_dL2=None, nwalkers=32, nsteps=1000, burn_in=500,
              show_progress=True):
    """
    Run MCMC for a single ratio type.

    Parameters:
    -----------
    rref : float
        Fixed log10(r_ref) for this run
    k_init, beta_init, log_C_Q_init : float
        Initial parameter guesses

    Returns:
    --------
    samples : array
        MCMC samples
    """
    if ratio_type == 'joint':
        ndim = 5  # k, beta, log_C_Q, offset_mg2, offset_si4
    else:
        ndim = 4  # k, beta, log_C_Q, offset

    # Starting positions around the initial guess
    if ratio_type == 'joint':
        pos = np.array([k_init, beta_init, log_C_Q_init, offset_init, offset_init2]) + 1e-3 * np.random.randn(nwalkers, ndim)
    else:
        pos = np.array([k_init, beta_init, log_C_Q_init, offset_init]) + 1e-3 * np.random.randn(nwalkers, ndim)

    # Create sampler
    if ratio_type == 'joint':
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior_joint,
            args=(observed_data, models, k_values, logQ_values, rref_values, beta_values, rref, log_4pi_dL2)
        )
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior_single,
            args=(observed_data[ratio_type], ratio_type, models, k_values, logQ_values,
                  rref_values, beta_values, rref, log_4pi_dL2)
        )

    if show_progress:
        print(f"Running MCMC with {nwalkers} walkers for {nsteps} steps...")
    sampler.run_mcmc(pos, nsteps, progress=show_progress)

    # Get samples (discard burn-in)
    samples = sampler.chain[:, burn_in:, :].reshape(-1, ndim)

    return samples


def plot_fit(observed_data, models, k_values, logQ_values, rref_values, beta_values, rref,
             params_dict, ratio_types=['mg2_c4', 'si4_c4'], filename=None,
             use_joint_params=False, rm_id=None, zoom_to_data=False):
    """
    Create fit plot with model uncertainty bands.

    Parameters:
    -----------
    rref : float
        Fixed log10(r_ref) used for this fit
    params_dict : dict
        Dictionary with keys like 'mg2_c4', 'si4_c4', 'joint'
        Each contains: 'k_median', 'k_std', 'log_C_Q_median', 'log_C_Q_std'
    use_joint_params : bool
        If True, use joint fit parameters for all ratios (for joint fit plots)
    rm_id : str
        RM ID for plot title
    """
    n_ratios = sum(1 for rt in ratio_types if observed_data[rt] is not None)
    if n_ratios == 0:
        return
    
    fig, axes = plt.subplots(1, n_ratios, figsize=(7*n_ratios, 6))
    if n_ratios == 1:
        axes = [axes]
    
    colors = {'mg2_c4': 'dodgerblue', 'si4_c4': 'deeppink', 'c3_c4': 'C2'}
    labels = {'mg2_c4': 'Mg II/C IV', 'si4_c4': '(Si IV+O IV)/C IV', 'c3_c4': 'C III]/C IV'}
    model_keys = {'mg2_c4': 'mg2c4', 'si4_c4': 'si4o4c4', 'c3_c4': None}
    
    # Get rm_id if not provided
    if rm_id is None:
        rm_id = observed_data.get('rm_id', 'unknown')
    
    ax_idx = 0
    for ratio_type in ratio_types:
        if observed_data[ratio_type] is None:
            continue
        
        ax = axes[ax_idx]
        color = colors.get(ratio_type, 'C0')
        label = labels.get(ratio_type, ratio_type)
        model_key = model_keys.get(ratio_type)
        
        if model_key is None:
            ax_idx += 1
            continue
        
        # Get parameters - if use_joint_params is True, always use joint params
        if use_joint_params and 'joint' in params_dict:
            params = params_dict['joint']
            # Get the appropriate offset for this ratio type
            if ratio_type == 'mg2_c4':
                offset_median = params.get('offset_mg2_median', 0.0)
            elif ratio_type == 'si4_c4':
                offset_median = params.get('offset_si4_median', 0.0)
            else:
                offset_median = 0.0
        elif ratio_type in params_dict:
            params = params_dict[ratio_type]
            offset_median = params.get('offset_median', 0.0)
        elif 'joint' in params_dict:
            params = params_dict['joint']
            # Get the appropriate offset for this ratio type
            if ratio_type == 'mg2_c4':
                offset_median = params.get('offset_mg2_median', 0.0)
            elif ratio_type == 'si4_c4':
                offset_median = params.get('offset_si4_median', 0.0)
            else:
                offset_median = 0.0
        else:
            ax_idx += 1
            continue
        
        k_median = params['k_median']
        k_std = params.get('k_std', 0.0)
        beta_median = params.get('beta_median', 0.5)  # Default to 0.5 if not found
        log_C_Q_median = params['log_C_Q_median']
        log_4pi_dL2_plot = params.get('log_4pi_dL2', None)
        
        f1350 = observed_data[ratio_type]['f1350']
        ratio = observed_data[ratio_type]['ratio']
        ratio_err = observed_data[ratio_type]['ratio_err']
        
        # Plot data
        ax.errorbar(f1350, ratio, yerr=ratio_err, fmt='o', ms=6,
                   alpha=0.7, label=r'$\rm Observed$', capsize=3, color=color,
                   elinewidth=1.5, capthick=1.5)
        
        # Use the model grid logQ values (54-56) and convert them to F1350
        # Step 1: Q -> photon_flux: photon_flux = Q / (4π * d_L²)  [photons/s/cm²]
        # Step 2: photon_flux -> F1350: F1350 = C_Q * photon_flux  [erg/cm²/s/Å]
        # So: F1350 = C_Q * Q / (4π * d_L²)
        # Taking log: log10(F1350) = log10(C_Q) + log10(Q) - log10(4π * d_L²)
        # So: log10(F1350) = log_C_Q + logQ - log_4pi_dL2
        # Therefore: F1350 = 10^(log_C_Q + logQ - log_4pi_dL2)
        # This means if logQ spans 54-56 (2 orders of mag), F1350 should also span 2 orders of mag
        if log_4pi_dL2_plot is not None:
            # Convert model logQ grid to F1350
            f1350_model = 10**(log_C_Q_median + logQ_values - log_4pi_dL2_plot)
            # Verify: logQ range should give 100x range in F1350
            expected_range = 10**(logQ_values.max() - logQ_values.min())  # Should be ~100 for 54-56
            actual_range = f1350_model.max() / f1350_model.min()
            print(f"  Model logQ: {logQ_values.min():.2f} to {logQ_values.max():.2f} (span: {logQ_values.max() - logQ_values.min():.2f} dex)")
            print(f"  Expected F1350 range: {expected_range:.1f}x, Actual: {actual_range:.1f}x")
        else:
            f1350_model = 10**(log_C_Q_median + logQ_values)
        
        # Plot best-fit model with uncertainty
        ratio_model_best = []
        ratio_model_best_err = []
        for logQ in logQ_values:
            model_result = ngp_interpolate(models, k_values, logQ_values, rref_values, beta_values, k_median, logQ, rref, beta_median)
            if model_result is not None:
                # Store model ratio without offset (will apply multiplicatively later)
                ratio_model_best.append(model_result[model_key])
                ratio_model_best_err.append(model_result[model_key + '_err'])
            else:
                ratio_model_best.append(np.nan)
                ratio_model_best_err.append(np.nan)
        
        ratio_model_best = np.array(ratio_model_best)
        ratio_model_best_err = np.array(ratio_model_best_err)
        
        # Sort by F1350 for plotting
        sort_idx = np.argsort(f1350_model)
        f1350_model_sorted = f1350_model[sort_idx]
        ratio_model_best_sorted = ratio_model_best[sort_idx]
        ratio_model_best_err_sorted = ratio_model_best_err[sort_idx]
        
        # Diagnostic: Check that model spans expected range
        f1350_range_model = f1350_model_sorted.max() / f1350_model_sorted.min()
        f1350_range_obs = f1350.max() / f1350.min()
        print(f"  Model F1350 range: {f1350_model_sorted.min():.2e} to {f1350_model_sorted.max():.2e} ({f1350_range_model:.1f}x)")
        print(f"  Observed F1350 range: {f1350.min():.2e} to {f1350.max():.2e} ({f1350_range_obs:.1f}x)")
        print(f"  Model should span ~100x (2 orders of mag in logQ from 54-56)")
        
        # Ensure model covers observed range - check coverage
        coverage_min = f1350_model_sorted.min() / f1350.min()  # < 1 means model extends below data
        coverage_max = f1350.max() / f1350_model_sorted.max()  # < 1 means model extends above data
        if coverage_min > 0.8 or coverage_max > 0.8:
            print(f"  Warning: Model F1350 range may not fully cover observed data range")
            print(f"    Model min / Observed min = {coverage_min:.2f} (should be < 0.8)")
            print(f"    Observed max / Model max = {coverage_max:.2f} (should be < 0.8)")
            print(f"    Consider adjusting log_C_Q prior to ensure full coverage")
        
        # Plot fiducial k models with colorbar FIRST (so best fit is on top)
        fiducial_k = np.linspace(k_values.min(), k_values.max(), 5)
        
        # Use plasma colormap for k values
        k_min_plot = min(fiducial_k)
        k_max_plot = max(fiducial_k)
        norm = plt.Normalize(vmin=k_min_plot, vmax=k_max_plot)
        cmap = plt.colormaps['plasma']
        
        for k_fid in fiducial_k:
            # Check if k_fid is in the grid
            if k_fid < k_values.min() or k_fid > k_values.max():
                continue
            
            # Get color from colormap
            fid_color = cmap(norm(k_fid))
            
            # Use same F1350_model grid (same C_Q and log_4pi_dL2)
            # Use best-fit gamma for fiducial k models
            ratio_model_fid = []
            for logQ in logQ_values:
                model_result = ngp_interpolate(models, k_values, logQ_values, rref_values, beta_values, k_fid, logQ, rref, beta_median)
                if model_result is not None:
                    # Store model ratio without offset (will apply multiplicatively later)
                    ratio_model_fid.append(model_result[model_key])
                else:
                    ratio_model_fid.append(np.nan)
            
            ratio_model_fid = np.array(ratio_model_fid)
            # Sort by F1350 and apply multiplicative offset
            ratio_model_fid_sorted = ratio_model_fid[sort_idx] * offset_median
            ax.plot(f1350_model_sorted, ratio_model_fid_sorted, '-', lw=2.5,
                   alpha=0.5, color=fid_color)
        
        # Plot best-fit model with uncertainty band LAST (so it's on top)
        # Apply multiplicative offset to model
        ratio_model_best_sorted_scaled = ratio_model_best_sorted * offset_median
        ratio_model_best_err_sorted_scaled = ratio_model_best_err_sorted * offset_median
        ax.plot(f1350_model_sorted, ratio_model_best_sorted_scaled, '-', lw=3.0,
               label=rf'$\rm Best\ fit:\ k = {k_median:.3f} \pm {k_std:.3f}$', color=color, zorder=10)
        ax.fill_between(f1350_model_sorted, 
                       ratio_model_best_sorted_scaled - ratio_model_best_err_sorted_scaled,
                       ratio_model_best_sorted_scaled + ratio_model_best_err_sorted_scaled,
                       alpha=0.2, color=color, zorder=9)
        
        # Add colorbar for k values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(r'$k\ \rm (metallicity\ gradient)$', rotation=270, labelpad=20)
        
        # Use log scale for x-axis (F1350) to match the logQ scale on top axis
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$F_{\lambda,1350}~\rm [erg~cm^{-2}~s^{-1}~\AA^{-1}]$')
        ax.set_ylabel(label)
        ax.set_title(f'{rm_id}: {label}')
        
        # Set x-axis limits and scale
        if zoom_to_data:
            # Linear axes, zoom to data range with 20% buffer
            ax.set_xscale('linear')
            ax.set_yscale('linear')
            f_range = f1350.max() - f1350.min()
            x_min = f1350.min() - 0.2 * f_range
            x_max = f1350.max() + 0.2 * f_range
            # Also set y limits from data with buffer
            r_range = ratio.max() - ratio.min()
            y_min = max(0, ratio.min() - 0.3 * r_range)
            y_max = ratio.max() + 0.3 * r_range
            ax.set_ylim(y_min, y_max)
        else:
            x_min = min(f1350.min(), f1350_model_sorted.min()) * 0.8
            x_max = max(f1350.max(), f1350_model_sorted.max()) * 1.2
        ax.set_xlim(x_min, x_max)
        
        # Add secondary x-axis for log Q
        # The model uses logQ values from 54 to 56
        # We convert these to F1350 using: F1350 = 10^(log_C_Q + logQ - log_4pi_dL2)
        # So the reverse is: logQ = log10(F1350) - log_C_Q + log_4pi_dL2
        # Since bottom axis is now log(F1350), the conversion is linear: logQ = log10(F1350) - log_C_Q + log_4pi_dL2
        
        def f1350_to_logQ(f1350_val):
            """Convert F1350 to log Q using the fitted C_Q."""
            # Since x-axis is log scale, f1350_val is already in log space (matplotlib handles this)
            # But the function receives actual F1350 values, so we need log10
            f1350_val = np.asarray(f1350_val)
            # Clip to positive values for log
            f1350_val = np.maximum(f1350_val, 1e-20)
            if log_4pi_dL2_plot is not None:
                result = np.log10(f1350_val) - log_C_Q_median + log_4pi_dL2_plot
            else:
                result = np.log10(f1350_val) - log_C_Q_median
            return result
        
        def logQ_to_f1350(logQ_val):
            """Convert log Q to F1350 using the fitted C_Q."""
            # Handle array or scalar
            logQ_val = np.asarray(logQ_val)
            if log_4pi_dL2_plot is not None:
                result = 10**(logQ_val + log_C_Q_median - log_4pi_dL2_plot)
            else:
                result = 10**(logQ_val + log_C_Q_median)
            return result
        
        # Use the model logQ values for the top axis
        # Map model logQ values to their corresponding F1350 positions
        f1350_axis_min = f1350_model_sorted[0]
        f1350_axis_max = f1350_model_sorted[-1]
        
        # Create secondary axis for logQ
        # Since bottom axis is log(F1350), the top axis shows logQ in linear scale
        secax = ax.secondary_xaxis('top', functions=(f1350_to_logQ, logQ_to_f1350))
        secax.set_xlabel(r'$\log Q~\rm [s^{-1}]$', labelpad=10)
        
        # Set tick locations for logQ (54-56 range) at regular intervals
        # Use the x_min and x_max we already calculated
        logQ_ticks = np.arange(54.0, 56.5, 0.5)  # 54, 54.5, 55, 55.5, 56
        f1350_ticks = logQ_to_f1350(logQ_ticks)
        
        # Filter ticks to be within the plotted range
        valid_ticks = (f1350_ticks >= x_min) & (f1350_ticks <= x_max)
        
        if np.any(valid_ticks):
            # Set ticks at the F1350 positions, but label them with logQ values
            secax.set_xticks(f1350_ticks[valid_ticks])
            secax.set_xticklabels([f'{x:.1f}' for x in logQ_ticks[valid_ticks]], fontsize=12)
        else:
            # Fallback: use model grid points that are visible
            valid_mask = (f1350_model_sorted >= x_min) & (f1350_model_sorted <= x_max)
            if np.any(valid_mask):
                logQ_valid = logQ_values[sort_idx][valid_mask]
                f1350_valid = f1350_model_sorted[valid_mask]
                # Use every Nth point to avoid too many ticks
                step = max(1, len(f1350_valid) // 5)
                secax.set_xticks(f1350_valid[::step])
                secax.set_xticklabels([f'{x:.1f}' for x in logQ_valid[::step]], fontsize=12)
        
        # Make sure the secondary axis doesn't use log scale
        secax.set_xscale('linear')
        
        # Ensure ticks are visible on top axis with same style as primary axis
        secax.tick_params(axis='x', which='major', labelsize=12, top=True, labeltop=True,
                         length=9, width=2, direction='in')
        secax.tick_params(axis='x', which='minor', top=True, length=4, width=2, direction='in')
        
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        ax.minorticks_on()
        # Set primary axis tick params - explicitly disable top ticks (secondary axis handles those)
        ax.tick_params(axis='both', which='major', top=False, labeltop=False, right=True, 
                      length=9, width=2, direction='in')
        ax.tick_params(axis='both', which='minor', top=False, right=True, 
                      length=4, width=2, direction='in')
        
        ax_idx += 1
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def discover_rref_values(model_file_pattern, beta_values):
    """
    Discover available log10(r_ref) values from existing model files.

    Parameters:
    -----------
    model_file_pattern : str
        Pattern with {rref} and {beta} placeholders
    beta_values : array
        Beta values to check

    Returns:
    --------
    rref_values : array
        Sorted array of log10(r_ref) values found in model files
    """
    import glob as globmod

    # Use first beta value to discover rref values
    beta_test = beta_values[0]
    # Build glob pattern: replace {rref:.2f} with wildcard, fix beta
    glob_pattern = model_file_pattern.format(rref=99.99, beta=beta_test)
    # Replace the dummy rref with wildcard
    glob_pattern = glob_pattern.replace("99.99", "*")
    files = globmod.glob(glob_pattern)

    rref_set = set()
    for f in files:
        # Extract rref from filename: ...rref{value}_beta...
        basename = os.path.basename(f)
        match = re.search(r'rref([\d.]+)_beta', basename)
        if match:
            rref_set.add(float(match.group(1)))

    return np.array(sorted(rref_set))


def main(rm_id=None, data_file=None, model_file=None, output_dir='mcmc_fits',
         plot_dir='mcmc_plots', fit_individual=False, show_progress=True,
         fit_mode='si4_only'):
    """Main fitting routine."""
    # Use defaults from top of file if not provided
    if rm_id is None:
        rm_id = globals().get('rm_id', 'rm002')
    if data_file is None:
        data_file = globals().get('observed_data_file', f"observed_line_ratio_data/{rm_id}_line_ratios.dat")

    print(f"Loading observed data for {rm_id}...")
    observed_data = load_observed_data(data_file)

    # Extract redshift and compute photon flux conversion
    redshift = observed_data.get('redshift')
    log_4pi_dL2 = None
    if redshift is not None:
        print(f"Redshift: z = {redshift:.6f}")
        log_4pi_dL2 = compute_photon_flux_conversion(redshift)
        print(f"log10(4pi * d_L^2): {log_4pi_dL2:.2f} cm^2")
    else:
        print("Warning: Redshift not found in data file, cannot compute photon flux conversion")

    # Check what data we have
    available_ratios = [k for k, v in observed_data.items() if v is not None and k != 'redshift']
    print(f"Available ratios: {available_ratios}")

    if 'mg2_c4' not in available_ratios and 'si4_c4' not in available_ratios:
        print("Error: Need at least mg2_c4 or si4_c4 data for fitting")
        return

    # Load model grid for (r_ref, beta) values
    beta_values_orig = globals().get('beta_values', np.arange(0.0, 1.05, 0.1))
    model_file_pat = globals().get('model_file_pattern',
        'mcmc_data/line_ratios_k_grid_gamma-1.2_rref{rref:.2f}_beta{beta:.2f}.dat')

    # Discover available r_ref values from files
    rref_values_discovered = discover_rref_values(model_file_pat, beta_values_orig)
    if len(rref_values_discovered) == 0:
        print("Error: No model files found! Run line_ratio_breathing_effect.py first.")
        return
    print(f"\nDiscovered rref values: {rref_values_discovered}")

    print(f"\nLoading model grids for beta values: {beta_values_orig}")
    models, k_values, logQ_values, rref_values_loaded, beta_values_loaded = load_model_grid(
        beta_values_orig, rref_values_discovered, model_file_pat)
    print(f"Loaded {len(rref_values_loaded)} rref values: {rref_values_loaded}")
    print(f"Loaded {len(beta_values_loaded)} beta values: {beta_values_loaded}")
    print(f"Original model grid: {len(k_values)} k values, {len(logQ_values)} logQ values")
    print(f"k range: [{k_values.min():.2f}, {k_values.max():.2f}]")
    print(f"logQ range: [{logQ_values.min():.2f}, {logQ_values.max():.2f}]")

    # Interpolate k and beta grids for smoother MCMC
    print(f"\nInterpolating k and beta grids...")
    models, k_values, beta_values_interp = interpolate_model_grid(
        models, k_values, logQ_values, rref_values_loaded, beta_values_loaded,
        n_k_interp=100, n_beta_interp=50)
    print(f"Interpolated model grid: {len(k_values)} k values, {len(beta_values_interp)} beta values, {len(logQ_values)} logQ values")

    # Use middle r_ref for fitting (fixed)
    rref_fit = rref_values_loaded[len(rref_values_loaded) // 2]
    print(f"\nUsing fixed rref = {rref_fit:.2f} for MCMC fitting")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # Initial parameter guess
    k_init = (k_values.min() + k_values.max()) / 2.0
    beta_init = 0.25  # Start in middle of [0, 0.5] prior range
    # Initial guess for log_C_Q
    if log_4pi_dL2 is not None:
        obs_f1350_all = []
        for ratio_type in ['mg2_c4', 'si4_c4']:
            if observed_data.get(ratio_type) is not None:
                obs_f1350_all.extend(observed_data[ratio_type]['f1350'])
        if obs_f1350_all:
            obs_f1350_all = np.array(obs_f1350_all)
            log_F1350_min = np.log10(obs_f1350_all.min())
            log_F1350_max = np.log10(obs_f1350_all.max())
            log_F1350_mid = (log_F1350_min + log_F1350_max) / 2.0
            logQ_mid = (logQ_values.min() + logQ_values.max()) / 2.0
            log_C_Q_init = log_F1350_mid + log_4pi_dL2 - logQ_mid
            log_C_Q_max_init = log_F1350_min - logQ_values.min() + log_4pi_dL2
            log_C_Q_min_init = log_F1350_max - logQ_values.max() + log_4pi_dL2
            log_C_Q_init = np.clip(log_C_Q_init, log_C_Q_min_init, log_C_Q_max_init)
            print(f"  Observed F1350 range: {obs_f1350_all.min():.2e} to {obs_f1350_all.max():.2e}")
            print(f"  Estimated log_C_Q from data: {log_C_Q_init:.2f}")
            print(f"  Prior range for log_C_Q: [{log_C_Q_min_init:.2f}, {log_C_Q_max_init:.2f}]")
        else:
            log_C_Q_init = -14.0
    else:
        log_C_Q_init = -14.0
    offset_init = 1.0

    print(f"\nInitial guess: k = {k_init:.3f}, beta = {beta_init:.2f}, log_C_Q = {log_C_Q_init:.2f}, offset = {offset_init:.3f}")

    # Store results
    params_dict = {}
    samples_dict = {}

    # Fit each ratio separately first (if requested)
    if fit_individual:
        for ratio_type in ['mg2_c4', 'si4_c4']:
            if observed_data[ratio_type] is None:
                continue

            print(f"\n{'='*60}")
            print(f"Fitting {ratio_type} separately...")
            print(f"{'='*60}")

            # Initial optimization
            print("Finding initial parameter estimates...")
            result = minimize(
                lambda p: -log_posterior_single(
                    p, observed_data[ratio_type], ratio_type, models, k_values, logQ_values,
                    rref_values_loaded, beta_values_interp, rref_fit, log_4pi_dL2
                ),
                [k_init, beta_init, log_C_Q_init, offset_init],
                method='Nelder-Mead',
                options={'maxiter': 1000}
            )

            if result.success:
                print(f"Optimization successful!")
                print(f"  k = {result.x[0]:.4f}")
                print(f"  beta = {result.x[1]:.4f}")
                print(f"  log_C_Q = {result.x[2]:.4f}")
                print(f"  offset = {result.x[3]:.4f}")
                k_opt, beta_opt, log_C_Q_opt, offset_opt = result.x
            else:
                print("Warning: Initial optimization did not converge, using default values")
                k_opt, beta_opt, log_C_Q_opt, offset_opt = k_init, beta_init, log_C_Q_init, offset_init

            # Run MCMC
            samples = run_mcmc(observed_data, ratio_type, models, k_values, logQ_values,
                              rref_values_loaded, beta_values_interp, rref_fit,
                              k_opt, beta_opt, log_C_Q_opt, offset_opt,
                              log_4pi_dL2=log_4pi_dL2, nwalkers=32, nsteps=1000, burn_in=500,
                              show_progress=show_progress)

            # Compute statistics
            k_median = np.median(samples[:, 0])
            k_std = np.std(samples[:, 0])
            beta_median = np.median(samples[:, 1])
            beta_std = np.std(samples[:, 1])
            log_C_Q_median = np.median(samples[:, 2])
            log_C_Q_std = np.std(samples[:, 2])
            offset_median = np.median(samples[:, 3])
            offset_std = np.std(samples[:, 3])

            params_dict[ratio_type] = {
                'k_median': k_median, 'k_std': k_std,
                'beta_median': beta_median, 'beta_std': beta_std,
                'log_C_Q_median': log_C_Q_median, 'log_C_Q_std': log_C_Q_std,
                'offset_median': offset_median, 'offset_std': offset_std,
                'log_4pi_dL2': log_4pi_dL2,
                'rref': rref_fit
            }
            samples_dict[ratio_type] = samples

            print(f"\nBest-fit parameters for {ratio_type}:")
            print(f"  k = {k_median:.4f} +/- {k_std:.4f}")
            print(f"  beta = {beta_median:.4f} +/- {beta_std:.4f}")
            print(f"  log_C_Q = {log_C_Q_median:.4f} +/- {log_C_Q_std:.4f}")
            print(f"  C_Q = {10**log_C_Q_median:.2e} erg cm^-2 s^-1 A^-1 per (photons/s/cm^2)")
            print(f"  offset = {offset_median:.4f} +/- {offset_std:.4f}")

            # Save samples
            np.savetxt(f"{output_dir}/{rm_id}_{ratio_type}_mcmc_samples.txt", samples,
                       header="k beta log_C_Q offset")

            # Create corner plot
            print("Creating corner plot...")
            if ratio_type == 'mg2_c4':
                offset_label = r'$A(\mathrm{MgII})$'
            elif ratio_type == 'si4_c4':
                offset_label = r'$A(\mathrm{SiIV})$'
            else:
                offset_label = r'$A$'

            fig = corner.corner(samples, labels=[r'$k$', r'$\beta$', r'$\log_{10} C_Q$', offset_label],
                               truths=[k_median, beta_median, log_C_Q_median, offset_median],
                               show_titles=True,
                               title_kwargs={"fontsize": 9},
                               label_kwargs={"fontsize": 10})
            for ax in fig.get_axes():
                ax.tick_params(labelsize=8)
                ax.xaxis.labelpad = 8
                ax.yaxis.labelpad = 8
            plt.savefig(f"{plot_dir}/{rm_id}_{ratio_type}_corner.png", dpi=300, bbox_inches='tight')
            plt.close()

    # Joint or si4_only fit
    samples_joint = None
    if fit_mode == 'joint' and 'mg2_c4' in available_ratios and 'si4_c4' in available_ratios:
        print(f"\n{'='*60}")
        print("Fitting both ratios jointly...")
        print(f"{'='*60}")

        # Use average of individual fits as starting point (if available)
        if 'mg2_c4' in params_dict and 'si4_c4' in params_dict:
            k_init_joint = (params_dict['mg2_c4']['k_median'] +
                          params_dict['si4_c4']['k_median']) / 2.0
            beta_init_joint = (params_dict['mg2_c4'].get('beta_median', beta_init) +
                              params_dict['si4_c4'].get('beta_median', beta_init)) / 2.0
            log_C_Q_init_joint = (params_dict['mg2_c4']['log_C_Q_median'] +
                                 params_dict['si4_c4']['log_C_Q_median']) / 2.0
            offset_mg2_init = params_dict['mg2_c4'].get('offset_median', 1.0)
            offset_si4_init = params_dict['si4_c4'].get('offset_median', 1.0)
        else:
            k_init_joint = k_init
            beta_init_joint = beta_init
            log_C_Q_init_joint = log_C_Q_init
            offset_mg2_init = offset_init
            offset_si4_init = offset_init

        # Initial optimization
        print("Finding initial parameter estimates...")
        result = minimize(
            lambda p: -log_posterior_joint(p, observed_data, models, k_values, logQ_values,
                                          rref_values_loaded, beta_values_interp, rref_fit, log_4pi_dL2),
            [k_init_joint, beta_init_joint, log_C_Q_init_joint, offset_mg2_init, offset_si4_init],
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )

        if result.success:
            print(f"Optimization successful!")
            k_opt_joint, beta_opt_joint, log_C_Q_opt_joint, offset_mg2_opt, offset_si4_opt = result.x
        else:
            k_opt_joint, beta_opt_joint, log_C_Q_opt_joint = k_init_joint, beta_init_joint, log_C_Q_init_joint
            offset_mg2_opt, offset_si4_opt = offset_mg2_init, offset_si4_init

        # Run MCMC
        samples_joint = run_mcmc(observed_data, 'joint', models, k_values, logQ_values,
                                rref_values_loaded, beta_values_interp, rref_fit,
                                k_opt_joint, beta_opt_joint, log_C_Q_opt_joint,
                                offset_mg2_opt, offset_si4_opt,
                                log_4pi_dL2=log_4pi_dL2, nwalkers=32, nsteps=1000, burn_in=500,
                                show_progress=show_progress)

    # Fit (SiIV+OIV)/CIV only
    elif fit_mode == 'si4_only' and 'si4_c4' in available_ratios:
        print(f"\n{'='*60}")
        print("Fitting (SiIV+OIV)/CIV only...")
        print(f"{'='*60}")

        k_init_joint = k_init
        beta_init_joint = beta_init
        log_C_Q_init_joint = log_C_Q_init
        offset_si4_init = offset_init

        # Initial optimization
        print("Finding initial parameter estimates...")
        result = minimize(
            lambda p: -log_posterior_single(p, observed_data['si4_c4'], 'si4_c4', models, k_values, logQ_values,
                                          rref_values_loaded, beta_values_interp, rref_fit, log_4pi_dL2),
            [k_init_joint, beta_init_joint, log_C_Q_init_joint, offset_si4_init],
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )

        if result.success:
            print(f"Optimization successful!")
            k_opt_joint, beta_opt_joint, log_C_Q_opt_joint, offset_si4_opt = result.x
        else:
            k_opt_joint, beta_opt_joint, log_C_Q_opt_joint = k_init_joint, beta_init_joint, log_C_Q_init_joint
            offset_si4_opt = offset_si4_init

        # Run MCMC (single ratio: 4 params)
        samples_joint = run_mcmc(observed_data, 'si4_c4', models, k_values, logQ_values,
                                rref_values_loaded, beta_values_interp, rref_fit,
                                k_opt_joint, beta_opt_joint, log_C_Q_opt_joint,
                                offset_si4_opt,
                                log_4pi_dL2=log_4pi_dL2, nwalkers=32, nsteps=1000, burn_in=500,
                                show_progress=show_progress)

    # Compute statistics and save plots (shared by both modes)
    if samples_joint is not None:
        k_median_joint = np.median(samples_joint[:, 0])
        k_std_joint = np.std(samples_joint[:, 0])
        beta_median_joint = np.median(samples_joint[:, 1])
        beta_std_joint = np.std(samples_joint[:, 1])
        log_C_Q_median_joint = np.median(samples_joint[:, 2])
        log_C_Q_std_joint = np.std(samples_joint[:, 2])

        if fit_mode == 'joint':
            offset_mg2_median = np.median(samples_joint[:, 3])
            offset_mg2_std = np.std(samples_joint[:, 3])
            offset_si4_median = np.median(samples_joint[:, 4])
            offset_si4_std = np.std(samples_joint[:, 4])
            params_dict['joint'] = {
                'k_median': k_median_joint, 'k_std': k_std_joint,
                'beta_median': beta_median_joint, 'beta_std': beta_std_joint,
                'log_C_Q_median': log_C_Q_median_joint, 'log_C_Q_std': log_C_Q_std_joint,
                'offset_mg2_median': offset_mg2_median, 'offset_mg2_std': offset_mg2_std,
                'offset_si4_median': offset_si4_median, 'offset_si4_std': offset_si4_std,
                'log_4pi_dL2': log_4pi_dL2,
                'rref': rref_fit
            }
            corner_labels = [r'$k$', r'$\beta$', r'$\log_{10} C_Q$',
                            r'$A(\mathrm{MgII})$', r'$A(\mathrm{SiIV})$']
            corner_truths = [k_median_joint, beta_median_joint, log_C_Q_median_joint,
                            offset_mg2_median, offset_si4_median]
            print(f"\nBest-fit parameters (joint):")
            print(f"  k = {k_median_joint:.4f} +/- {k_std_joint:.4f}")
            print(f"  beta = {beta_median_joint:.4f} +/- {beta_std_joint:.4f}")
            print(f"  log_C_Q = {log_C_Q_median_joint:.4f} +/- {log_C_Q_std_joint:.4f}")
            print(f"  C_Q = {10**log_C_Q_median_joint:.2e} erg cm^-2 s^-1 A^-1 per (photons/s/cm^2)")
            print(f"  offset_mg2 = {offset_mg2_median:.4f} +/- {offset_mg2_std:.4f}")
            print(f"  offset_si4 = {offset_si4_median:.4f} +/- {offset_si4_std:.4f}")
            sample_header = "k beta log_C_Q offset_mg2 offset_si4"
        else:  # si4_only
            offset_si4_median = np.median(samples_joint[:, 3])
            offset_si4_std = np.std(samples_joint[:, 3])
            params_dict['joint'] = {
                'k_median': k_median_joint, 'k_std': k_std_joint,
                'beta_median': beta_median_joint, 'beta_std': beta_std_joint,
                'log_C_Q_median': log_C_Q_median_joint, 'log_C_Q_std': log_C_Q_std_joint,
                'offset_si4_median': offset_si4_median, 'offset_si4_std': offset_si4_std,
                'log_4pi_dL2': log_4pi_dL2,
                'rref': rref_fit
            }
            corner_labels = [r'$k$', r'$\beta$', r'$\log_{10} C_Q$', r'$A(\mathrm{SiIV})$']
            corner_truths = [k_median_joint, beta_median_joint, log_C_Q_median_joint,
                            offset_si4_median]
            print(f"\nBest-fit parameters (si4_only):")
            print(f"  k = {k_median_joint:.4f} +/- {k_std_joint:.4f}")
            print(f"  beta = {beta_median_joint:.4f} +/- {beta_std_joint:.4f}")
            print(f"  log_C_Q = {log_C_Q_median_joint:.4f} +/- {log_C_Q_std_joint:.4f}")
            print(f"  C_Q = {10**log_C_Q_median_joint:.2e} erg cm^-2 s^-1 A^-1 per (photons/s/cm^2)")
            print(f"  offset_si4 = {offset_si4_median:.4f} +/- {offset_si4_std:.4f}")
            sample_header = "k beta log_C_Q offset_si4"

        samples_dict['joint'] = samples_joint

        # Save samples
        np.savetxt(f"{output_dir}/{rm_id}_joint_mcmc_samples.txt", samples_joint,
                   header=sample_header)

        # Create corner plot
        print("Creating corner plot...")
        fig = corner.corner(samples_joint,
                           labels=corner_labels,
                           truths=corner_truths,
                           show_titles=True,
                           title_kwargs={"fontsize": 9},
                           label_kwargs={"fontsize": 10})
        for ax in fig.get_axes():
            ax.tick_params(labelsize=8)
            ax.xaxis.labelpad = 8
            ax.yaxis.labelpad = 8
        plt.savefig(f"{plot_dir}/{rm_id}_joint_corner.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Create fit plots
    print("\nCreating fit plots...")

    # Plot individual fits (if available)
    if fit_individual:
        for ratio_type in ['mg2_c4', 'si4_c4']:
            if ratio_type in params_dict:
                plot_fit(observed_data, models, k_values, logQ_values,
                        rref_values_loaded, beta_values_interp, rref_fit,
                        params_dict, ratio_types=[ratio_type],
                        filename=f"{plot_dir}/{rm_id}_{ratio_type}_fit.png",
                        rm_id=rm_id)

    # Plot fit (joint or si4_only)
    if 'joint' in params_dict:
        if fit_mode == 'si4_only':
            plot_ratio_types = ['si4_c4']
        else:
            plot_ratio_types = ['mg2_c4', 'si4_c4']
        plot_fit(observed_data, models, k_values, logQ_values,
                rref_values_loaded, beta_values_interp, rref_fit,
                params_dict, ratio_types=plot_ratio_types,
                use_joint_params=True,
                filename=f"{plot_dir}/{rm_id}_joint_fit.png",
                rm_id=rm_id)
        # Zoomed version (x range set by data)
        plot_fit(observed_data, models, k_values, logQ_values,
                rref_values_loaded, beta_values_interp, rref_fit,
                params_dict, ratio_types=plot_ratio_types,
                use_joint_params=True,
                filename=f"{plot_dir}/{rm_id}_joint_fit_zoom.png",
                rm_id=rm_id, zoom_to_data=True)

    # Save best-fit parameters
    with open(f"{output_dir}/{rm_id}_bestfit.txt", 'w') as f:
        f.write(f"# Best-fit parameters for {rm_id}\n")
        f.write(f"# gamma = {gamma_fixed} (fixed, Korista & Goad 2019)\n")
        f.write(f"# rref = {rref_fit:.2f} (fixed log10 r_ref)\n\n")
        for fit_type, params in params_dict.items():
            f.write(f"# {fit_type.upper()} FIT\n")
            f.write(f"# k (metallicity gradient)\n")
            f.write(f"k_{fit_type} = {params['k_median']:.6f} +/- {params['k_std']:.6f}\n")
            f.write(f"# beta (breathing factor)\n")
            f.write(f"beta_{fit_type} = {params.get('beta_median', 0.5):.6f} +/- {params.get('beta_std', 0.0):.6f}\n")
            f.write(f"# log10(C_Q) where F1350 = C_Q * photon_flux\n")
            f.write(f"log_C_Q_{fit_type} = {params['log_C_Q_median']:.6f} +/- {params['log_C_Q_std']:.6f}\n")
            f.write(f"C_Q_{fit_type} = {10**params['log_C_Q_median']:.6e} erg cm^-2 s^-1 A^-1 per (photons/s/cm^2)\n")
            if fit_type == 'joint' and 'offset_mg2_median' in params:
                f.write(f"# offset (multiplicative offset between observed and model ratios)\n")
                f.write(f"offset_mg2_{fit_type} = {params['offset_mg2_median']:.6f} +/- {params['offset_mg2_std']:.6f}\n")
                f.write(f"offset_si4_{fit_type} = {params['offset_si4_median']:.6f} +/- {params['offset_si4_std']:.6f}\n")
            elif fit_type == 'joint' and 'offset_si4_median' in params:
                f.write(f"# offset (multiplicative offset between observed and model ratios)\n")
                f.write(f"offset_si4_{fit_type} = {params['offset_si4_median']:.6f} +/- {params['offset_si4_std']:.6f}\n")
            else:
                f.write(f"# offset (multiplicative offset between observed and model ratios)\n")
                f.write(f"offset_{fit_type} = {params['offset_median']:.6f} +/- {params['offset_std']:.6f}\n")
            f.write("\n")

    print(f"\nBest-fit parameters saved to {output_dir}/{rm_id}_bestfit.txt")
    print("\nDone!")


def fit_single_object_wrapper(args):
    """
    Wrapper function for parallel processing of single object fits.

    Parameters:
    -----------
    args : tuple
        (rm_id, data_file, output_dir, plot_dir, fit_individual)

    Returns:
    --------
    result : dict
        {'rm_id': rm_id, 'success': bool, 'error': str or None}
    """
    rm_id, data_file, output_dir, plot_dir, fit_individual, fit_mode = args
    try:
        # Suppress per-object output in batch mode
        import io, sys
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for parallel
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            main(rm_id=rm_id, data_file=data_file,
                 output_dir=output_dir, plot_dir=plot_dir,
                 fit_individual=fit_individual, show_progress=False,
                 fit_mode=fit_mode)
        finally:
            sys.stdout = old_stdout
        return {'rm_id': rm_id, 'success': True, 'error': None}
    except Exception as e:
        import traceback, sys
        sys.stdout = sys.__stdout__
        error_msg = f"{e}\n{traceback.format_exc()}"
        return {'rm_id': rm_id, 'success': False, 'error': error_msg}


def fit_all_objects(data_dir='observed_line_ratio_data',
                    output_dir='mcmc_fits', plot_dir='mcmc_plots',
                    fit_individual=False, n_cores=None, fit_mode='si4_only'):
    """
    Fit all objects in the observed_line_ratio_data directory.

    Parameters:
    -----------
    data_dir : str
        Directory containing observed line ratio data files
    output_dir : str
        Directory for MCMC fit results
    plot_dir : str
        Directory for plots
    fit_individual : bool
        Whether to also fit individual ratios separately
    n_cores : int or None
        Number of CPU cores to use for parallel processing.
        If None, uses all available cores.
    """
    import glob
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    # Find all data files
    pattern = os.path.join(data_dir, 'rm*_line_ratios.dat')
    data_files = sorted(glob.glob(pattern))

    if len(data_files) == 0:
        print(f"No data files found in {data_dir}")
        return

    print(f"Found {len(data_files)} data files to fit")

    # Extract rm_id from each file and prepare arguments
    fit_args = []
    for data_file in data_files:
        basename = os.path.basename(data_file)
        match = re.match(r'(rm\d+)_line_ratios\.dat', basename)
        if match:
            rm_id = match.group(1)
            fit_args.append((rm_id, data_file, output_dir,
                            plot_dir, fit_individual, fit_mode))
        else:
            print(f"Warning: Could not extract rm_id from {basename}, skipping")

    if len(fit_args) == 0:
        print("No valid data files to fit")
        return

    # Determine number of cores
    if n_cores is None:
        n_cores = cpu_count()
    n_cores = min(n_cores, len(fit_args), cpu_count())

    print(f"Using {n_cores} CPU cores for parallel processing")
    print(f"Fitting {len(fit_args)} objects...")

    # Run fits in parallel with progress bar
    if n_cores > 1:
        with Pool(processes=n_cores) as pool:
            results = list(tqdm(pool.imap_unordered(fit_single_object_wrapper, fit_args),
                                total=len(fit_args), desc="Fitting objects"))
    else:
        results = [fit_single_object_wrapper(args) for args in tqdm(fit_args, desc="Fitting objects")]

    # Report results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n{'='*80}")
    print(f"Batch fitting complete!")
    print(f"  Successful: {len(successful)}/{len(results)}")
    if len(failed) > 0:
        print(f"  Failed: {len(failed)}")
        for r in failed:
            print(f"    {r['rm_id']}: {r['error'].split(chr(10))[0]}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Fit line ratios using MCMC')
    parser.add_argument('--batch', action='store_true',
                       help='Fit all objects in observed_line_ratio_data directory')
    parser.add_argument('--ncores', type=int, default=None,
                       help='Number of CPU cores for parallel processing (default: all available)')
    parser.add_argument('--fit-individual', action='store_true',
                       help='Also fit individual ratios separately (in addition to joint fit)')
    parser.add_argument('--fit-mode', choices=['joint', 'si4_only'], default='joint',
                       help='Fitting mode: joint (MgII+SiIV) or si4_only (default: joint)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for fits (default: joint_fits or nagao_ratio_fits)')
    parser.add_argument('--plot-dir', type=str, default=None,
                       help='Output directory for plots (default: joint_plots or nagao_ratio_plots)')

    args = parser.parse_args()

    # Set default output dirs based on fit mode
    if args.output_dir is None:
        output_dir = 'joint_fits' if args.fit_mode == 'joint' else 'nagao_ratio_fits'
    else:
        output_dir = args.output_dir
    if args.plot_dir is None:
        plot_dir = 'joint_plots' if args.fit_mode == 'joint' else 'nagao_ratio_plots'
    else:
        plot_dir = args.plot_dir

    if args.batch:
        fit_all_objects(fit_individual=args.fit_individual, n_cores=args.ncores,
                       fit_mode=args.fit_mode, output_dir=output_dir, plot_dir=plot_dir)
    else:
        main(fit_individual=args.fit_individual, fit_mode=args.fit_mode,
             output_dir=output_dir, plot_dir=plot_dir)
