"""Smoke test: short MCMC + plot_fit across the four flag combinations
(default / --extra_sig / --fit_r_ref / both).  Does NOT touch the real fit
outputs.  Run from repo root with the project env:

    ~/miniconda3/envs/pypeit/bin/python alpha/_test_band_plot.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import fit_line_ratios as F

RM = 'rm002'
DATA_FILE = f'data/alpha/observed_line_ratio_data/{RM}_line_ratios.dat'
OLD_SAMPLES = f'fits/alpha/joint_fits/{RM}_joint_mcmc_samples.txt'
MODEL_PAT = 'data/alpha/mcmc_data/line_ratios_k_grid_gamma-1.2_rref{rref:.2f}_beta{beta:.2f}.dat'

obs = F.load_observed_data(DATA_FILE)
log_4pi_dL2 = F.compute_photon_flux_conversion(obs.get('redshift'))

beta_values_orig = np.arange(0.0, 1.05, 0.1)
rref_disc = F.discover_rref_values(MODEL_PAT, beta_values_orig)
models, k_values, logQ_values, rref_loaded, beta_loaded = F.load_model_grid(
    beta_values_orig, rref_disc, MODEL_PAT)
models, k_values, beta_interp = F.interpolate_model_grid(
    models, k_values, logQ_values, rref_loaded, beta_loaded, n_k_interp=100, n_beta_interp=50)
rref_fit = rref_loaded[len(rref_loaded) // 2]
print(f"rref grid: {list(rref_loaded)}, rref_fit (middle) = {rref_fit:.2f}")

old = np.loadtxt(OLD_SAMPLES)
k0, b0, lcq0, om0, os0 = np.median(old, axis=0)

for use_extra_sig in (False, True):
    for use_fit_r_ref in (False, True):
        tag = ("extrasig" if use_extra_sig else "nosig") + "_" + ("fitrref" if use_fit_r_ref else "fixrref")
        S = F.run_mcmc(obs, 'joint', models, k_values, logQ_values, rref_loaded, beta_interp, rref_fit,
                       k0, b0, lcq0, om0, os0,
                       log_f_init=np.log(0.1), log_r_ref_init=float(rref_fit),
                       use_extra_sig=use_extra_sig, use_fit_r_ref=use_fit_r_ref,
                       log_4pi_dL2=log_4pi_dL2, nwalkers=24, nsteps=120, burn_in=40,
                       show_progress=False)
        exp_ndim = 5 + int(use_extra_sig) + int(use_fit_r_ref)
        assert S.shape[1] == exp_ndim, f"{tag}: ndim {S.shape[1]} != {exp_ndim}"
        msg = f"[{tag}] chain {S.shape}"
        i_col = 5
        params = {
            'k_median': float(np.median(S[:, 0])), 'k_std': float(np.std(S[:, 0])),
            'beta_median': float(np.median(S[:, 1])),
            'log_C_Q_median': float(np.median(S[:, 2])), 'log_C_Q_std': float(np.std(S[:, 2])),
            'offset_mg2_median': float(np.median(S[:, 3])),
            'offset_si4_median': float(np.median(S[:, 4])),
            'log_4pi_dL2': log_4pi_dL2,
        }
        if use_extra_sig:
            lf = float(np.median(S[:, i_col]))
            params.update({'log_f_median': lf, 'log_f_std': float(np.std(S[:, i_col])),
                           'f_median': float(np.exp(lf))})
            msg += f"  f={np.exp(lf):.3f}"
            i_col += 1
        if use_fit_r_ref:
            lr = float(np.median(S[:, i_col]))
            params.update({'log_r_ref_median': lr, 'log_r_ref_std': float(np.std(S[:, i_col]))})
            msg += f"  log_r_ref={lr:.3f}"
            i_col += 1
        print(msg)
        out = f'plots/alpha/joint_plots/_TEST_{tag}.png'
        F.plot_fit(obs, models, k_values, logQ_values, rref_loaded, beta_interp, rref_fit,
                   {'joint': params}, {'joint': S}, ratio_types=['mg2_c4', 'si4_c4'],
                   use_joint_params=True, filename=out, rm_id=RM)
        print(('OK ' if os.path.exists(out) else 'FAIL ') + out)
