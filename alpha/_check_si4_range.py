"""Range of log[(SiIV+OIV)/CIV] in the a2/b2/c2 heatmap panels.  Scratch."""
import numpy as np
from alpha.loc_gradient_fit import make_interpolators, model_line_ratios

interps = make_interpolators()
logQ = np.linspace(53.5, 56.0, 70)


def rng(label, vals):
    v = np.array(vals)
    print(f"{label:28s}: {v.min():+.3f} .. {v.max():+.3f}   (span {v.max()-v.min():.3f} dex)")


rng('a2  vary k (p=0.35,lrr=17)',
    [np.log10(model_line_ratios(interps, logQ, kk, 0.35, log_r_ref=17.0, z_norm=3.0)[0]['(SiIV+OIV)/CIV'])
     for kk in np.linspace(-0.6, 0.6, 60)])
rng('b2  vary p (k=0,lrr=17)',
    [np.log10(model_line_ratios(interps, logQ, 0.0, pp, log_r_ref=17.0, z_norm=3.0)[0]['(SiIV+OIV)/CIV'])
     for pp in np.linspace(0.0, 0.5, 60)])
rng('c2  vary log_r_ref (k=0,p=0.4)',
    [np.log10(model_line_ratios(interps, logQ, 0.0, 0.4, log_r_ref=rr, z_norm=3.0)[0]['(SiIV+OIV)/CIV'])
     for rr in np.linspace(17.0, 18.0, 60)])
