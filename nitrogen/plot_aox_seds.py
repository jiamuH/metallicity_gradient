#!/usr/bin/env python3
"""
Plot the incident SEDs from the alpha_ox Cloudy grid alongside the Nagao
'strong bump' interpolated SED, for visual comparison.

All SEDs are normalized at 2500 A (the conventional alpha_ox reference
wavelength) so that the X-ray-to-UV slope differences set by alpha_ox are
directly visible as fan-out in the X-ray.

Run from repo root:
    python3 nitrogen/plot_aox_seds.py
"""

import glob
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.rcParams.update({
    'text.usetex': True,
    'axes.linewidth': 2,
    'font.family': 'serif',
    'font.weight': 'heavy',
    'font.size': 16,
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm} \usepackage{amsmath} \boldmath'

AOX_GRID_DIR = '/Users/jiamuh/c23.01/my_models/aox_nitrogen_grid'
NAGAO_CON = ('/Users/jiamuh/c23.01/my_models/m1n9phi18/'
             'strong_n9_phi_18_N25_m1_v100_SED.con')
OUT_PATH = Path('plots/nitrogen/aox_seds_comparison.png')

# Plot range (Angstroms)
WAVE_MIN = 1e-1     # 0.1 A ~ 100 keV
WAVE_MAX = 1e5      # 100,000 A ~ near-IR
NORM_WAVE = 2500.0  # alpha_ox reference wavelength


def read_first_block(filename):
    """Read columns (wavelength_A, incident_nuFnu) from the FIRST grid block
    of a Cloudy .con file. Stops at the first GRID_DELIMIT marker; if there
    are no markers, reads to EOF.
    """
    waves, fnu = [], []
    with open(filename) as f:
        for line in f:
            # NOTE: check GRID_DELIMIT before the '#' skip — the delimiter
            # line itself starts with '#'.
            if 'GRID_DELIMIT' in line:
                break
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                w = float(parts[0])
                inc = float(parts[1])
            except ValueError:
                continue
            waves.append(w)
            fnu.append(inc)
    return np.array(waves), np.array(fnu)


def normalize_at(wavelengths, values, w_norm):
    """Normalize so that interp(w_norm) == 1. Assumes wavelengths are sorted."""
    order = np.argsort(wavelengths)
    w_sorted = wavelengths[order]
    v_sorted = values[order]
    # Use log interpolation (more robust over a SED span)
    mask = (v_sorted > 0) & np.isfinite(v_sorted)
    log_v = np.interp(np.log10(w_norm),
                      np.log10(w_sorted[mask]),
                      np.log10(v_sorted[mask]))
    norm = 10 ** log_v
    if norm <= 0:
        return values
    return values / norm


def main():
    # Load alpha_ox SEDs
    con_files = sorted(glob.glob(os.path.join(AOX_GRID_DIR, 'aox*_Z.con')))
    if not con_files:
        raise FileNotFoundError(f'No .con files in {AOX_GRID_DIR}')

    aox_seds = []
    for cf in con_files:
        m = re.search(r'aox(-?\d+\.\d+)_Z\.con', os.path.basename(cf))
        if not m:
            continue
        aox = float(m.group(1))
        w, inc = read_first_block(cf)
        if w.size == 0:
            print(f'  WARNING: empty {cf}')
            continue
        inc_n = normalize_at(w, inc, NORM_WAVE)
        aox_seds.append((aox, w, inc_n))
    aox_seds.sort(key=lambda t: t[0])
    print(f'Loaded {len(aox_seds)} alpha_ox SEDs')

    # Load Nagao SED
    nagao_w, nagao_inc = read_first_block(NAGAO_CON)
    nagao_inc_n = normalize_at(nagao_w, nagao_inc, NORM_WAVE)
    print(f'Loaded Nagao SED: {len(nagao_w)} points')

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.cm.viridis
    aox_arr = np.array([t[0] for t in aox_seds])
    norm = Normalize(vmin=aox_arr.min(), vmax=aox_arr.max())

    for aox, w, inc_n in aox_seds:
        mask = (w >= WAVE_MIN) & (w <= WAVE_MAX) & (inc_n > 0)
        ax.plot(w[mask], inc_n[mask], '-', lw=2.5,
                color=cmap(norm(aox)), alpha=0.9)

    nmask = (nagao_w >= WAVE_MIN) & (nagao_w <= WAVE_MAX) & (nagao_inc_n > 0)
    ax.plot(nagao_w[nmask], nagao_inc_n[nmask], 'k--', lw=3.5,
            label='Nagao strong bump', zorder=10)

    # Mark normalization wavelength + key ionization edges
    ax.axvline(NORM_WAVE, color='gray', ls=':', lw=1, alpha=0.6)
    ax.text(NORM_WAVE * 1.05, ax.get_ylim()[1] * 0.5 if False else 1e-3,
            r'$2500\,\AA$', color='gray', fontsize=12,
            rotation=90, va='bottom', ha='left')

    # Vertical reference lines for ionization edges
    edges = [
        (912.0,  r'$\rm H\,I$'),
        (228.0,  r'$\rm He\,II$'),
        (161.0,  r'$\rm N\,V$'),
        (12.4,   r'$\rm 1\,keV$'),
    ]
    for w_e, lbl in edges:
        ax.axvline(w_e, color='gray', ls=':', lw=0.8, alpha=0.5)
        ax.text(w_e, 5e2, lbl, color='gray', fontsize=11,
                rotation=90, va='top', ha='right')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(WAVE_MIN, WAVE_MAX)
    ax.set_xlabel(r'$\lambda\ [\AA]$')
    ax.set_ylabel(r'$\nu F_\nu\ /\ \nu F_\nu(2500\,\AA)$')
    ax.minorticks_on()
    ax.tick_params(top=True, right=True, axis='both', which='major',
                   length=8, width=1.5, direction='in')
    ax.tick_params(top=True, right=True, axis='both', which='minor',
                   length=4, width=1, direction='in')

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'$\alpha_{\rm ox}$')
    cbar.ax.tick_params(direction='in', which='both')

    ax.legend(loc='lower left', frameon=False, fontsize=14)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT_PATH}')
    plt.close(fig)


if __name__ == '__main__':
    main()
