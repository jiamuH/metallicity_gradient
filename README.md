# metallicity_gradient

Analysis tools for inferring nitrogen abundance and metallicity gradients in
AGN broad-line regions (BLRs) from Cloudy photoionization models and SDSS-RM
spectra.

Migrated from `cloudy_notebooks/`.

## Layout

```
metallicity_gradient/
├── nitrogen/                       Nitrogen abundance scripts + notebooks
│   ├── compute_nitrogen_abundance.py
│   ├── nitrogen_abundance_vs_r.dat
│   ├── Cloudy_LOC_nitrogen_vturb.ipynb
│   ├── Cloudy_LOC_nitrogen_old.ipynb
│   ├── Nagao_Ratio_Nitrogen_LOC*.ipynb
│   └── Nagao_Ratio_NIII]_NIV]*.ipynb
├── alpha/                          Alpha-element metallicity-gradient pipeline
│   ├── extract_line_ratios.py
│   ├── line_ratio_breathing_effect.py
│   ├── fit_line_ratios.py
│   ├── plot_mcmc_bestfit_distributions.py
│   ├── plot_grad_results.py
│   └── Cloudy_LOC_metal_*.ipynb, Cloudy_LOC_for_CLAGN_metal.ipynb
├── data/alpha/                     Inputs
│   ├── mcmc_data/                  Cloudy model grids
│   └── observed_line_ratio_data/   rmNNN_line_ratios.dat from SDSS-RM
├── fits/alpha/                     Fit results (text)
│   ├── mcmc_fits/
│   ├── nagao_ratio_fits/
│   ├── joint_fits/
│   ├── fit_results/
│   ├── zgard_fits/
│   └── QA_and_results/
└── plots/
    ├── nitrogen/
    └── alpha/
        ├── mcmc_plots/
        ├── nagao_ratio_plots/
        ├── joint_plots/
        ├── observed_line_ratio_plots/
        ├── grad_results_plots/
        ├── line_ratio_plots/
        ├── model_line_ratio_plots/
        ├── mgii_civ_plots_by_rref/
        ├── siiv_oiv_ratio_plots_by_rref/
        ├── fit_plots/
        ├── final_model_plots/
        └── summary/                 Loose root-level summary PNGs
```

## Running scripts

All hardcoded paths in the `.py` files are relative to the repo root, so
**run scripts from `metallicity_gradient/` (not from inside `alpha/` or
`nitrogen/`)**:

```bash
cd /Users/jiamuh/python/metallicity_gradient
python3 alpha/plot_grad_results.py
python3 alpha/plot_mcmc_bestfit_distributions.py
python3 alpha/fit_line_ratios.py --batch --fit-mode joint
python3 nitrogen/compute_nitrogen_abundance.py
```

## External dependencies

Scripts read Cloudy / SDSS-RM data from outside this repo:

- `/Users/jiamuh/c23.01/my_models/loc_metal/` — alpha-element metal grid
- `/Users/jiamuh/c23.01/my_models/singlezone_nitrogen_series/` — nitrogen grid
- `/Users/jiamuh/sdssrm/` — SDSS-RM `_t.dat` / `_c1350.dat` flux files
  (read by `extract_line_ratios.py`)

## Install

```
pip install -r requirements.txt
```

## Typical pipeline

1. `alpha/extract_line_ratios.py` → `data/alpha/observed_line_ratio_data/`
2. `alpha/line_ratio_breathing_effect.py` → `data/alpha/mcmc_data/`
3. `alpha/fit_line_ratios.py` (per-object or `--batch`) → `fits/alpha/mcmc_fits/` (or `joint_fits` / `nagao_ratio_fits`)
4. `alpha/plot_mcmc_bestfit_distributions.py` → `plots/alpha/mcmc_plots/`
5. `alpha/plot_grad_results.py` → `plots/alpha/grad_results_plots/`
6. `nitrogen/compute_nitrogen_abundance.py` → `nitrogen/nitrogen_abundance_vs_r.dat`

Outputs (`.png`, `.dat`, `.fits`, etc.) are ignored by `.gitignore` except
`requirements.txt`.
