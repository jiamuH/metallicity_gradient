#!/usr/bin/env python3
"""Nested-sampling (dynesty) run of the SiIV-only mock fit.

Nested sampling maps a multimodal posterior with honest mode weights, which
emcee (stretch or DE moves) could not deliver for the nearly information-free
(SiIV+OIV)/CIV-only likelihood.  Runs on the mock249grad103 data, prints the
evidence and the posterior mass in each log r_ref regime, and saves an
equal-weight sample file plus a truth-marked corner plot.

Run: python3 -m alpha._dynesty_si4 [si4] [mg2] [--nlive=N] [--dlogz=X]
(default: si4 only, nlive=500, dlogz=0.1)
Outputs:
  fits/alpha/mock_recovery_si4/mock249grad103_dynesty_samples.txt
  plots/alpha/mock_recovery_si4/corner/mock249grad103_dynesty_corner.png
"""
import sys

import numpy as np

from alpha.loc_gradient_fit import (PARAM_BOUNDS, load_object_data,
                                    load_model_grid_interp, log_likelihood,
                                    param_latex)

NAME = 'mock249grad103'
TRUTH = [-0.5, 0.40, 17.2, np.log10(3.0), np.log(0.15)]
BOUNDS = [PARAM_BOUNDS[n] for n in
          ('k', 'p', 'log_r_ref', 'log10_Z_norm', 'log_f')]
LO = np.array([b[0] for b in BOUNDS])
WID = np.array([b[1] - b[0] for b in BOUNDS])


def run(ratio, nlive=500, dlogz=0.1):
    import dynesty
    data, _ = load_object_data(NAME)
    d = {ratio: data[ratio]}
    gi = load_model_grid_interp()

    def loglike(theta):
        return log_likelihood(theta, d, gi)

    def ptform(u):
        return LO + WID * u

    sampler = dynesty.NestedSampler(loglike, ptform, ndim=5, nlive=nlive,
                                    bound='multi', sample='rwalk')
    sampler.run_nested(dlogz=dlogz, print_progress=True)
    res = sampler.results
    samp = res.samples_equal()
    tag = ratio.split('_')[0]
    out = f'fits/alpha/mock_recovery_{tag}/{NAME}_dynesty_samples.txt'
    np.savetxt(out, samp, header='k p log_r_ref log10_Z_norm log_f')
    print(f'\n{ratio}: logZ = {res.logz[-1]:.2f} +/- {res.logzerr[-1]:.2f}, '
          f'{len(samp)} equal-weight samples -> {out}')
    frac = float((samp[:, 2] > 18.0).mean())
    print(f'posterior mass with log r_ref > 18: {frac:.2f}')
    for i, n in enumerate(('k', 'p', 'log_r_ref', 'log10_Z_norm', 'ln_f')):
        lo, mid, hi = np.percentile(samp[:, i], [16, 50, 84])
        print(f'  {n:14s} {mid:+.3f} [{lo:+.3f}, {hi:+.3f}]   '
              f'truth {TRUTH[i]:+.3f}')

    import corner
    import matplotlib.pyplot as plt
    fig = corner.corner(samp, labels=param_latex(use_free_a=False),
                        truths=TRUTH, truth_color='orangered',
                        show_titles=True, bins=40, smooth=1.0, smooth1d=1.0,
                        title_kwargs={'fontsize': 16},
                        label_kwargs={'fontsize': 20})
    for ax in fig.get_axes():
        ax.tick_params(labelsize=13)
    png = f'plots/alpha/mock_recovery_{tag}/corner/{NAME}_dynesty_corner.png'
    fig.savefig(png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'corner: {png}')


if __name__ == '__main__':
    asked = [a for a in sys.argv[1:] if a in ('si4', 'mg2')]
    ratios = [f'{a}_c4' for a in (asked or ['si4'])]
    kw = {}
    for a in sys.argv[1:]:
        if a.startswith('--nlive='):
            kw['nlive'] = int(a.split('=')[1])
        if a.startswith('--dlogz='):
            kw['dlogz'] = float(a.split('=')[1])
    for r in ratios:
        run(r, **kw)
