#!/usr/bin/env python3
"""Injection-recovery test of the LOC + partial-breathing gradient pipeline.

Two mock objects are built on the real epoch sampling of RM 249 (same F1350
values, same quoted ratio errors, same redshift):

  mock249flat : k = 0.0,  p = 0.48  (flat profile, near-full breathing)
  mock249grad : k = -0.5, p = 0.40  (real gradient, partial breathing)

both at Z_norm = 3 Zsun, log r_ref = 17.2, and intrinsic scatter f = 0.15.
Mock ratios are drawn as R = m + N(0, sqrt(sig^2 + (f m)^2)) with m the model
ratio from the same precomputed grid the fits use, so the generative model
matches the fitted one exactly.  The mock .dat files are written into the
observed-data directory (clearly named mock249*) so the unmodified pipeline
reader picks them up.

Generate the mock data files (fast):
    python3 alpha/mock_gradient_recovery.py
Run the two fits with the paper settings (28 walkers x 2500 steps, no-free-A):
    python3 alpha/mock_gradient_recovery.py --fit
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from alpha.loc_gradient_fit import (_OBS_DATA_DIR, f1350_to_logQ,
                                    load_model_grid_interp, fit_object)

SRC_ID = 'rm249'
TRUTHS = {
    'mock249flat': dict(k=0.0, p=0.48, log_r_ref=17.2,
                        lz=np.log10(3.0), f=0.15),
    'mock249grad': dict(k=-0.5, p=0.40, log_r_ref=17.2,
                        lz=np.log10(3.0), f=0.15),
    'mock249pos':  dict(k=+0.5, p=0.40, log_r_ref=17.2,
                        lz=np.log10(3.0), f=0.15),
    'mock249mid':  dict(k=-0.2, p=0.30, log_r_ref=17.2,
                        lz=np.log10(3.0), f=0.15),
    'mock249mid18': dict(k=-0.2, p=0.30, log_r_ref=18.0,
                         lz=np.log10(3.0), f=0.15),
    'mock249mid185': dict(k=-0.2, p=0.30, log_r_ref=18.5,
                          lz=np.log10(3.0), f=0.15),
}
# extra noise realizations of the steep-gradient case (same truth as
# mock249grad, independent seeds) to test whether its shallow recovered k
# is a fluctuation or a systematic of the k-p degeneracy
for _s in (101, 102, 103, 104):
    TRUTHS[f'mock249grad{_s}'] = dict(k=-0.5, p=0.40, log_r_ref=17.2,
                                      lz=np.log10(3.0), f=0.15, seed=_s)
COL = {'mg2_c4': 0, 'si4_c4': 1}
SEED = 42


def make_mocks():
    src = os.path.join(_OBS_DATA_DIR, f'{SRC_ID}_line_ratios.dat')
    with open(src) as fh:
        lines = fh.readlines()
    import re
    z = None
    for line in lines:
        if line.startswith('#'):
            m = re.search(r'z\s*=\s*([\d.]+)', line)
            if m:
                z = float(m.group(1))
    if z is None:
        raise ValueError(f'no redshift in {src}')

    interp = load_model_grid_interp()
    shared_rng = np.random.default_rng(SEED)
    for name, t in TRUTHS.items():
        # entries with their own seed get a dedicated generator; the others
        # keep drawing from the shared sequential stream so that the files
        # already generated (and fitted) stay bit-identical
        rng = (np.random.default_rng(t['seed']) if 'seed' in t
               else shared_rng)
        out_lines = []
        for line in lines:
            p = line.split()
            if line.startswith('#') or not line.strip() or len(p) < 5:
                out_lines.append(line)
                continue
            f1350, sig, typ = float(p[0]), float(p[2]), p[4]
            if typ not in COL or f1350 <= 1e-19:
                out_lines.append(line)
                continue
            lq = float(f1350_to_logQ(np.array([f1350]), z)[0])
            m = float(interp(np.array([[lq, t['k'], t['p'],
                                        t['log_r_ref'], t['lz']]]))[0, COL[typ]])
            r = m + rng.normal(0.0, np.hypot(sig, t['f'] * m))
            p[1] = f'{max(r, 1e-4):.6f}'
            out_lines.append('  '.join(p) + '\n')
        dst = os.path.join(_OBS_DATA_DIR, f'{name}_line_ratios.dat')
        with open(dst, 'w') as fh:
            fh.writelines(out_lines)
        print(f'wrote {dst}  (truth: k={t["k"]:+.2f}, p={t["p"]:.2f}, '
              f'log_r_ref={t["log_r_ref"]}, Z_norm={10**t["lz"]:.1f}, '
              f'f={t["f"]})')


def _truth_vector(t):
    return [t['k'], t['p'], t['log_r_ref'], t['lz'], np.log(t['f'])]


def replot_corners(only=None, out_dir='fits/alpha/mock_recovery',
                   plot_dir='plots/alpha/mock_recovery'):
    """Redraw the mock corner plots from the saved chains with the injected
    truth marked in orangered (the fit pipeline's own corners mark the
    posterior medians instead)."""
    import corner
    import matplotlib.pyplot as plt
    from alpha.loc_gradient_fit import param_latex
    for name, t in TRUTHS.items():
        if only and name not in only:
            continue
        path = os.path.join(out_dir, f'{name}_loc_mcmc_samples.txt')
        if not os.path.exists(path):
            continue
        samples = np.loadtxt(path)
        fig = corner.corner(samples, labels=param_latex(use_free_a=False),
                            truths=_truth_vector(t), truth_color='orangered',
                            show_titles=True, bins=24, smooth=1.0, smooth1d=1.0,
                            title_kwargs={'fontsize': 16},
                            label_kwargs={'fontsize': 20})
        for ax in fig.get_axes():
            ax.tick_params(labelsize=13)
        out = os.path.join(plot_dir, 'corner', f'{name}_loc_corner.png')
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'replotted {out} (orangered = injected truth)')


def combine_corners(name='mock249grad103',
                    out='plots/alpha/mock_recovery/corner/%s_corner_compare.png',
                    dynesty_singles=False):
    """Overlay the joint, MgII-only, and SiIV-only posteriors of one mock in a
    single corner plot (contours only, no scatter), with the injected truth.
    dynesty_singles=True takes the single-ratio posteriors from the nested
    sampling runs (honest mode weights) instead of the emcee chains."""
    import corner
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from alpha.loc_gradient_fit import param_latex

    runs = [('fits/alpha/mock_recovery', 'k',
             r'$\rm joint~(both~ratios)$'),
            ('fits/alpha/mock_recovery_mg2', 'dodgerblue',
             r'$\rm Mg\,II/C\,IV~only$'),
            ('fits/alpha/mock_recovery_si4', 'deeppink',
             r'$\rm (Si\,IV+O\,IV])/C\,IV~only$')]
    samples = []
    for d, _, _ in runs:
        fn = f'{name}_loc_mcmc_samples.txt'
        if dynesty_singles and d.endswith(('_mg2', '_si4')):
            fn = f'{name}_dynesty_samples.txt'
        samples.append(np.loadtxt(os.path.join(d, fn)))
    ndim = samples[0].shape[1]
    rng_ = [(min(s[:, i].min() for s in samples),
             max(s[:, i].max() for s in samples)) for i in range(ndim)]

    from matplotlib.colors import to_rgba
    labels = [r'$k$', r'$p$', r'$\log r_{\rm ref}~[\rm cm]$',
              r'$\log Z_{\rm ref}~[Z_{\odot}]$', r'$\ln f$']
    levels = (1 - np.exp(-0.5), 1 - np.exp(-2.0))
    fig = None
    # draw the joint posterior last so its black contours sit on top
    order = [1, 2, 0]
    for s, (_, color, _) in [(samples[i], runs[i]) for i in order]:
        # three fill bands (outside 2-sigma, 2-1 sigma, inside 1-sigma);
        # the outside band must stay transparent or it floods the panel
        bands = ['none', to_rgba(color, 0.30), to_rgba(color, 0.55)]
        fig = corner.corner(s, fig=fig, range=rng_, color=color,
                            weights=np.full(len(s), 1.0 / len(s)),
                            levels=levels, bins=40, smooth=2.8, smooth1d=1.5,
                            plot_datapoints=False, plot_density=False,
                            fill_contours=True,
                            contour_kwargs={'linewidths': 1.5},
                            contourf_kwargs={'colors': bands},
                            hist_kwargs={'lw': 2.0},
                            labels=labels,
                            label_kwargs={'fontsize': 20})
    # fill under the 1-D marginals (corner draws them as bare lines); recompute
    # the same smoothed histograms corner plotted
    from scipy.ndimage import gaussian_filter1d
    axes = np.array(fig.axes).reshape(ndim, ndim)
    for idx in order:
        s, (_, color, _) = samples[idx], runs[idx]
        for i in range(ndim):
            n, b = np.histogram(s[:, i], bins=40, range=rng_[i],
                                weights=np.full(len(s), 1.0 / len(s)))
            n = gaussian_filter1d(n.astype(float), 1.5)
            x = 0.5 * (b[1:] + b[:-1])
            axes[i, i].fill_between(x, 0, n, step='mid',
                                    color=to_rgba(color, 0.30), lw=0)

    t = TRUTHS[name]
    corner.overplot_lines(fig, _truth_vector(t), color='orangered', lw=1.8)
    for ax in fig.get_axes():
        ax.tick_params(labelsize=13)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(fc=to_rgba(c, 0.35), ec=c, lw=2.0, label=l)
                        for _, c, l in runs]
               + [Line2D([], [], color='orangered', lw=2.5,
                         label=r'$\rm Truth$')],
               loc='upper right', fontsize=20, frameon=False,
               bbox_to_anchor=(0.98, 0.98))
    path = out % name
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {path}')


def run_fits(only=None, ratio_types=None, n_steps=2500, burn_in=1000,
             n_walkers=28, de_moves=False):
    """ratio_types=None fits both ratios jointly (the paper pipeline);
    ['mg2_c4'] or ['si4_c4'] fits a single ratio, with outputs kept in
    separate mock_recovery_mg2 / mock_recovery_si4 directories.
    de_moves=True swaps the stretch move for a differential-evolution
    mixture, which hops between separated modes far more effectively."""
    tag = '' if not ratio_types else '_' + ratio_types[0].split('_')[0]
    out_dir = f'fits/alpha/mock_recovery{tag}'
    plot_dir = f'plots/alpha/mock_recovery{tag}'
    for name, t in TRUTHS.items():
        if only and name not in only:
            continue
        res = fit_object(name, n_walkers=n_walkers, n_steps=n_steps,
                         burn_in=burn_in, out_dir=out_dir, plot_dir=plot_dir,
                         use_free_a=False, ratio_types=ratio_types,
                         de_moves=de_moves)
        replot_corners(only=[name], out_dir=out_dir, plot_dir=plot_dir)
        truth = _truth_vector(t)
        print(f'\n{name}: recovered (16/50/84 pct) vs truth')
        for n_, lo, mid, hi, tr in zip(res['names'], res['p16'], res['p50'],
                                       res['p84'], truth):
            flag = '' if lo - 0.1 <= tr <= hi + 0.1 else '  <-- off'
            print(f'  {n_:14s} {mid:+.3f} [{lo:+.3f}, {hi:+.3f}]   '
                  f'truth {tr:+.3f}{flag}')


if __name__ == '__main__':
    make_mocks()
    names = [a for a in sys.argv[1:] if not a.startswith('-')]
    kw = {}
    for a in sys.argv[1:]:
        if a.startswith('--steps='):
            kw['n_steps'] = int(a.split('=')[1])
        if a.startswith('--burn='):
            kw['burn_in'] = int(a.split('=')[1])
        if a.startswith('--walkers='):
            kw['n_walkers'] = int(a.split('=')[1])
    if '--de' in sys.argv:
        kw['de_moves'] = True
    if '--fit' in sys.argv:
        run_fits(only=names or None, **kw)
    if '--fit-mg2' in sys.argv:
        run_fits(only=names or None, ratio_types=['mg2_c4'], **kw)
    if '--fit-si4' in sys.argv:
        run_fits(only=names or None, ratio_types=['si4_c4'], **kw)
