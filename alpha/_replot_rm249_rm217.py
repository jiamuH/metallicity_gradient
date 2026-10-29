"""Replot the RM 249 and RM 217 model-vs-data figures (paper Figures 7 and 13) from the saved
chains of the paper fit (no free line normalisations, Q_ref = 1e58)."""
from alpha.loc_gradient_fit import replot_object

for rm in ['rm249', 'rm217']:
    replot_object(rm, out_dir='fits/alpha/loc_gradient_noA_Qref58', plot_dir='plots/alpha/loc_gradient_noA_Qref58')
