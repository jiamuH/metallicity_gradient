#!/usr/bin/env python3
"""Diagnostic: walker traces + convergence stats for the mock single-ratio fits.

The saved flat sample files come from sampler.get_chain(discard=burn_in,
flat=True), which flattens (n_steps, n_walkers, ndim) row-major, so they can
be reshaped back into per-walker traces.  Plots parameter-vs-step traces per
walker and prints the integrated autocorrelation time and the fraction of
walkers in each r_ref mode, for the SiIV-only, MgII-only, and joint fits of
mock249grad103.

Run: python3 -m alpha._check_chain_convergence
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

import emcee

# optional argument: walker count of the single-ratio runs (joint stays 28)
N_WALKERS_SINGLE = int(sys.argv[1]) if len(sys.argv) > 1 else 28
N_WALKERS = 28
NAMES = ['k', 'p', r'log r_ref', r'log Z_norm', r'ln f']

RUNS = [('si4', 'fits/alpha/mock_recovery_si4'),
        ('mg2', 'fits/alpha/mock_recovery_mg2'),
        ('joint', 'fits/alpha/mock_recovery')]

for tag, d in RUNS:
    nw = N_WALKERS if tag == 'joint' else N_WALKERS_SINGLE
    flat = np.loadtxt(f'{d}/mock249grad103_loc_mcmc_samples.txt')
    chain = flat.reshape(-1, nw, flat.shape[1])          # (steps, walkers, ndim)
    n_steps = chain.shape[0]
    print(f'\n--- {tag}: {n_steps} post-burn-in steps x {nw} walkers ---')
    try:
        tau = emcee.autocorr.integrated_time(chain, quiet=True)
        print('  autocorr time tau:',
              '  '.join(f'{n}={t:.0f}' for n, t in zip(NAMES, tau)),
              f'  (chain length / max tau = {n_steps / tau.max():.1f})')
    except Exception as e:
        print(f'  autocorr failed: {e}')
    # mode occupancy in log r_ref (modes separated at 18.0), first vs last 20%
    rr = chain[:, :, 2]
    for label, sl in (('first 20%', slice(0, n_steps // 5)),
                      ('last 20%', slice(-n_steps // 5, None))):
        frac = float((rr[sl] > 18.0).mean())
        print(f'  fraction of samples with log r_ref > 18 ({label}): {frac:.2f}')
    # per-walker mode switches
    w_mode = rr > 18.0
    switches = int((w_mode[1:] != w_mode[:-1]).sum())
    print(f'  total walker mode crossings at log r_ref = 18: {switches}')

    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    for i, (ax, nm) in enumerate(zip(axes, NAMES)):
        ax.plot(chain[:, :, i], lw=0.4, alpha=0.5)
        ax.set_ylabel(nm, fontsize=13)
        ax.tick_params(direction='in', top=True, right=True)
    axes[-1].set_xlabel('step (post burn-in)', fontsize=13)
    out = f'plots/alpha/mock_recovery_{tag}/mock249grad103_chain_trace.png' \
        if tag != 'joint' else \
        'plots/alpha/mock_recovery/mock249grad103_chain_trace.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  trace plot: {out}')
