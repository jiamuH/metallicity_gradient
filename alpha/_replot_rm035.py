"""Re-make the rm035 joint-fit plot from the *saved* MCMC samples (no re-run),
to check the central line now sits inside the credible band.
Run from repo root:  ~/miniconda3/envs/pypeit/bin/python alpha/_replot_rm035.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import fit_line_ratios as F

RM = 'rm035'
DATA_FILE = f'data/alpha/observed_line_ratio_data/{RM}_line_ratios.dat'
SAMPLES   = f'fits/alpha/joint_fits/{RM}_joint_mcmc_samples.txt'   # 5 cols: k beta log_C_Q offset_mg2 offset_si4
MODEL_PAT = 'data/alpha/mcmc_data/line_ratios_k_grid_gamma-1.2_rref{rref:.2f}_beta{beta:.2f}.dat'

obs = F.load_observed_data(DATA_FILE)
log_4pi_dL2 = F.compute_photon_flux_conversion(obs.get('redshift'))
beta_values_orig = np.arange(0.0, 1.05, 0.1)
rref_disc = F.discover_rref_values(MODEL_PAT, beta_values_orig)
models, k_values, logQ_values, rref_loaded, beta_loaded = F.load_model_grid(beta_values_orig, rref_disc, MODEL_PAT)
models, k_values, beta_interp = F.interpolate_model_grid(models, k_values, logQ_values, rref_loaded, beta_loaded,
                                                         n_k_interp=100, n_beta_interp=50)
rref_fit = rref_loaded[len(rref_loaded) // 2]

S = np.loadtxt(SAMPLES)
params = {
    'k_median': float(np.median(S[:, 0])), 'k_std': float(np.std(S[:, 0])),
    'beta_median': float(np.median(S[:, 1])),
    'log_C_Q_median': float(np.median(S[:, 2])), 'log_C_Q_std': float(np.std(S[:, 2])),
    'offset_mg2_median': float(np.median(S[:, 3])),
    'offset_si4_median': float(np.median(S[:, 4])),
    'log_4pi_dL2': log_4pi_dL2,
}
F.plot_fit(obs, models, k_values, logQ_values, rref_loaded, beta_interp, rref_fit,
           {'joint': params}, {'joint': S}, ratio_types=['mg2_c4', 'si4_c4'], use_joint_params=True,
           filename='plots/alpha/joint_plots/rm035_joint_fit.png', rm_id=RM)
F.plot_fit(obs, models, k_values, logQ_values, rref_loaded, beta_interp, rref_fit,
           {'joint': params}, {'joint': S}, ratio_types=['mg2_c4', 'si4_c4'], use_joint_params=True,
           filename='plots/alpha/joint_plots/rm035_joint_fit_zoom.png', rm_id=RM, zoom_to_data=True)
print("replotted rm035")
