# metallicity_gradient

Analysis tools for inferring nitrogen abundance and metallicity gradients in
AGN broad-line regions (BLRs) from Cloudy photoionization models and SDSS-RM
spectra.

Migrated from `cloudy_notebooks/` so the metallicity / nitrogen abundance
work lives in its own repo.

## Contents

### Nitrogen abundance

- `compute_nitrogen_abundance.py` — derive `(N/H)/(N/H)_sun` vs. BLR radius
  from Cloudy singlezone nitrogen models; writes `nitrogen_abundance_vs_r.dat`.
- `nitrogen_abundance_vs_r.dat` — output table.
- `Cloudy_LOC_nitrogen_vturb.ipynb`, `Cloudy_LOC_nitrogen_old.ipynb` — LOC
  nitrogen exploratory notebooks.
- `Nagao_Ratio_Nitrogen_LOC*.ipynb`, `Nagao_Ratio_NIII]_NIV]*.ipynb` —
  earlier Nagao-style nitrogen ratio analyses.

### Metallicity gradient

- `extract_line_ratios.py` — pulls Mg II/C IV, C III]/C IV, Si IV/C IV
  ratios and `F_1350` from SDSS-RM `_t.dat` / `_c1350.dat` flux files into
  `observed_line_ratio_data/`.
- `line_ratio_breathing_effect.py` — generates the Cloudy 3D model grid
  collapsed over density and convolved with a Gaussian breathing window;
  outputs `mcmc_data/line_ratios_k_grid_*.dat` and root-level summary PNGs.
- `fit_line_ratios.py` — MCMC fit (emcee) of the metallicity-gradient
  parameter `k` and the `Q → F_1350` conversion; writes results to
  `mcmc_fits/` and corner plots to `nagao_ratio_plots/`.
- `plot_mcmc_bestfit_distributions.py` — collects per-object best-fit
  parameters and plots the inferred `Z(r)` profiles into `mcmc_plots/`.
- `plot_grad_results.py` — plots best-fit `k` versus mean `F_1350`.
- `Cloudy_LOC_metal_series*.ipynb`, `Cloudy_LOC_metal_n9phi18.ipynb`,
  `Cloudy_LOC_for_CLAGN_metal.ipynb` — Cloudy metal grid notebooks that
  feed the model grid used by `fit_line_ratios.py`.

## Directory layout

| Directory | Role |
|---|---|
| `mcmc_data/` | Cloudy model grids (input to MCMC) |
| `observed_line_ratio_data/` | Observed `rmNNN_line_ratios.dat` files |
| `observed_line_ratio_plots/` | Per-object ratio-vs-flux plots |
| `mcmc_fits/`, `mcmc_plots/` | MCMC outputs and per-object plots |
| `nagao_ratio_fits/`, `nagao_ratio_plots/` | Nagao-style fits/plots |
| `joint_fits/`, `joint_plots/` | Joint Mg II + Si IV fits and plots |
| `grad_results_plots/` | `k` vs. `F_1350` summary plots |
| `line_ratio_plots/`, `model_line_ratio_plots/` | Cloudy line-ratio plots |
| `mgii_civ_plots_by_rref/`, `siiv_oiv_ratio_plots_by_rref/` | Per-`r_ref` ratio plots |
| `fit_results/`, `fit_plots/` | Earlier scalar fits |
| `final_model_plots/`, `QA_and_results/`, `zgard_fits/` | Misc. results |

Output `.png`, `.dat`, `.fits`, etc. are ignored by `.gitignore` (except
`requirements.txt`).

## External dependencies

The scripts read Cloudy model output from outside this repo:

- `/Users/jiamuh/c23.01/my_models/loc_metal/` — main metal grid
- `/Users/jiamuh/c23.01/my_models/singlezone_nitrogen_series/` — nitrogen grid

SDSS-RM flux files are read from `/Users/jiamuh/sdssrm/` by
`extract_line_ratios.py`.

## Install

```
pip install -r requirements.txt
```

## Typical pipeline

1. `extract_line_ratios.py` → `observed_line_ratio_data/`
2. `line_ratio_breathing_effect.py` → `mcmc_data/`
3. `fit_line_ratios.py` (set `rm_id` per object) → `mcmc_fits/`
4. `plot_mcmc_bestfit_distributions.py` → `mcmc_plots/`
5. `plot_grad_results.py` → `grad_results_plots/`
6. `compute_nitrogen_abundance.py` → `nitrogen_abundance_vs_r.dat`
