"""Fit rm035 with the new --fit_r_ref machinery (default ON) and no extra-variance.
Writes to the standard joint_fits folders.  Run from repo root:

    python3 alpha/_run_rm035_var.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import fit_line_ratios as F

RM = 'rm035'
F.main(rm_id=RM,
       data_file=f'data/alpha/observed_line_ratio_data/{RM}_line_ratios.dat',
       output_dir='fits/alpha/joint_fits',
       plot_dir='plots/alpha/joint_plots',
       fit_individual=False,
       fit_mode='joint',
       use_extra_sig=False,
       use_fit_r_ref=True,
       show_progress=True)
print("DONE rm035 -> fits/alpha/joint_fits , plots/alpha/joint_plots")
