#!/usr/bin/env python3
"""
Two-panel (upper-lower) version of the microturbulence figure:
nitrogen line ratios NIII]/NIV]/NV over CIV (top) and over HeII (bottom)
as a function of v_turb, with and without turbulent dissipation heating.

Replaces the two side-by-side single-panel figures
(Cloudy_nitrogen_civ_vturb.png, Cloudy_nitrogen_he2_vturb.png) made in
nitrogen/Cloudy_LOC_nitrogen_vturb.ipynb.

Run from repo root:
    python3 nitrogen/plot_vturb_2panel.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

plt.rcParams.update({
    'text.usetex': True,
    'axes.linewidth': 2,
    'font.family': 'serif',
    'font.weight': 'heavy',
    'font.size': 20,
})
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{bm} \boldmath'

# Custom colormap (same palette as Cloudy_LOC_nitrogen_vturb.ipynb)
_CUSTOM_COLORS = [
    (0.15, 0.08, 0.40),
    (0.294, 0.161, 0.569),
    (0.529, 0.173, 0.635),
    (0.753, 0.212, 0.616),
    (0.918, 0.310, 0.533),
    (0.980, 0.471, 0.463),
    (0.965, 0.663, 0.478),
    (0.929, 0.851, 0.639),
]
cmap = LinearSegmentedColormap.from_list('custom_cmap', _CUSTOM_COLORS, N=256)
colors_custom = [cmap(i) for i in np.linspace(0.1, 0.9, 3)]  # N3, N4, N5

# ----------------------------------------------------------------------
# Observed composite (Van den Berk et al. 2001)
NV1240_obs, NV1240_err = 2.461, 0.189
NIV1718_obs, NIV1718_err = 0.258, 0.027
NIII1750_obs, NIII1750_err = 0.382, 0.021
CIV1549_obs, CIV1549_err = 25.29, 0.11
HeII1640_obs, HeII1640_err = 0.52, 0.03


def obs_ratio(num, num_err, den, den_err):
    r = num / den
    re_ = r * np.sqrt((num_err / num) ** 2 + (den_err / den) ** 2)
    return r, re_


R_N3C4, R_N3C4_err = obs_ratio(NIII1750_obs, NIII1750_err, CIV1549_obs, CIV1549_err)
R_N4C4, R_N4C4_err = obs_ratio(NIV1718_obs, NIV1718_err, CIV1549_obs, CIV1549_err)
R_N5C4, R_N5C4_err = obs_ratio(NV1240_obs, NV1240_err, CIV1549_obs, CIV1549_err)
R_N3He2, R_N3He2_err = obs_ratio(NIII1750_obs, NIII1750_err, HeII1640_obs, HeII1640_err)
R_N4He2, R_N4He2_err = obs_ratio(NIV1718_obs, NIV1718_err, HeII1640_obs, HeII1640_err)
R_N5He2, R_N5He2_err = obs_ratio(NV1240_obs, NV1240_err, HeII1640_obs, HeII1640_err)

# ----------------------------------------------------------------------
# Cloudy vturb series (single zone, hden=10, phi=19, N=23, 1 Zsun)
v_grid = np.arange(1, 1500, 50)
BASE = '/Users/jiamuh/c23.01/my_models/vturb_series/'
FILE_DISS = os.path.join(
    BASE, 'vmax1500',
    'strong_n10_phi19_N23_m1_vary_vturb_LineList_BLR_Fe2_flux.txt')
FILE_NODISS = os.path.join(
    BASE, 'vmax1500_no_dissipate',
    'strong_n10_phi19_N23_m1_vary_vturb_nodiss_LineList_BLR_Fe2_flux.txt')

LINE_NAMES = {
    'HeII': 'he 2 1640.41A',
    'C4':   'blnd 1549.00A',
    'N5':   'blnd 1240.00A',
    'N4':   'blnd 1486.00A',
    'N3':   'blnd 1750.00A',
}


def read_lines(path):
    df = pd.read_csv(path, sep='\t', header=0, comment=None)
    df = df[~df.iloc[:, 0].astype(str).str.startswith('#')]
    return {key: df[col].values for key, col in LINE_NAMES.items()}


diss = read_lines(FILE_DISS)
nodiss = read_lines(FILE_NODISS)

line_labels = {'N3': r'$\rm NIII]$', 'N4': r'$\rm NIV]$', 'N5': r'$\rm NV$'}

# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 11), sharex=True)
fig.subplots_adjust(hspace=0.04)

panels = [
    # (axis, denominator key, observed bands, y label)
    (ax1, 'C4',
     [(R_N3C4, R_N3C4_err), (R_N4C4, R_N4C4_err), (R_N5C4, R_N5C4_err)],
     r'$\rm N_X / C~IV~\lambda1549$'),
    (ax2, 'HeII',
     [(R_N3He2, R_N3He2_err), (R_N4He2, R_N4He2_err), (R_N5He2, R_N5He2_err)],
     r'$\rm N_X / HeII~\lambda1640$'),
]

for ax, den, bands, ylabel in panels:
    for i, key in enumerate(['N3', 'N4', 'N5']):
        ax.plot(v_grid, diss[key] / diss[den], color=colors_custom[i],
                lw=3, ls='-', solid_capstyle='round')
        ax.plot(v_grid, nodiss[key] / nodiss[den], color=colors_custom[i],
                lw=3, ls='--', path_effects=[pe.Stroke(capstyle='round')])
    for i, (R, Rerr) in enumerate(bands):
        ax.axhspan(R - Rerr, R + Rerr, color=colors_custom[i], alpha=0.4)
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', top=True, right=True,
                   length=9, width=2, direction='in', pad=5)
    ax.tick_params(axis='both', which='minor', top=True, right=True,
                   length=5, width=2, direction='in')

ax2.set_xlabel(r'$v_{\rm turb}~[\rm km~s^{-1}]$')

# Legends on the top panel only
line_handles = [Line2D([], [], color=colors_custom[i], lw=3) for i in range(3)]
leg1 = ax1.legend(line_handles, list(line_labels.values()),
                  fontsize=16, loc='upper left')
style_handles = [Line2D([], [], color='black', lw=3, ls='-'),
                 Line2D([], [], color='black', lw=3, ls='--')]
leg2 = ax1.legend(style_handles,
                  [r'$\rm with~dissipation$', r'$\rm no~dissipation$'],
                  fontsize=16, loc='upper right')
ax1.add_artist(leg1)

OUT = 'plots/nitrogen/Cloudy_nitrogen_vturb_2panel.png'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'-> wrote {OUT}')
plt.close(fig)
