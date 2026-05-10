#!/usr/bin/env python3
"""
Plot line/CIV ratios as a function of alpha_ox for the BLR-typical single-zone
Cloudy grid in /Users/jiamuh/c23.01/my_models/aox_nitrogen_grid/.

Science argument:
    For a fixed N/H, the LOC model predicts certain NV/CIV, NIV]/CIV,
    NIII]/CIV ratios; observationally NV is much stronger than NIV] and NIII]
    relative to those predictions, so the inferred N/H from NV alone is much
    higher than from NIV] or NIII].

    A potential alternative explanation is a harder ionizing SED (less negative
    alpha_ox), which selectively boosts NV (it requires hard photons to ionize
    N IV -> N V) without similarly boosting the lower-ionization N lines.

    This figure tests that hypothesis by showing the N V/C IV, N IV]/C IV,
    N III]/C IV ratios as a function of alpha_ox at fixed solar metallicity
    (top row), alongside the low-ionization checks MgII/CIV, CIII]/CIV,
    CII]/CIV (bottom row). Observed BLR composite ratios (Van den Berk 2001)
    are overlaid as horizontal bands.

    The argument: shifting alpha_ox to bring NV/CIV to the observed value
    pushes the low-ionization ratios away from their observed values -> SED
    hardening cannot resolve the N anomaly, so enhanced N/H is the right
    interpretation.

Run from the repo root:
    python3 nitrogen/plot_aox_line_ratios.py
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

GRID_DIR = "/Users/jiamuh/c23.01/my_models/aox_nitrogen_grid"
OUT_DIR = Path("plots/nitrogen")
OUT_PATH = OUT_DIR / "aox_line_ratios.png"

# Column labels (Cloudy line labels in the LineList header)
LINE_COLS = {
    'NV':    'blnd 1240.00A',
    'NIV':   'blnd 1486.00A',
    'NIII':  'blnd 1750.00A',
    'CIV':   'blnd 1549.00A',
    'CIII':  'blnd 1909.00A',
    'CII':   'blnd 2326.00A',
    'MgII':  'blnd 2798.00A',
}

# Reference metallicity slice to highlight (log Z/Zsun)
LOG_Z_REF = 0.0

# Observed BLR composite line ratios (relative to CIV = 100) from
# Van den Berk et al. 2001 (Table 2; SDSS quasar composite). Values quoted
# as Wlam(line)/Wlam(CIV); we treat them as proxies for the line/CIV flux
# ratio for this comparison plot. Errors are propagated as fractional sums.
# (NV1240, NIV]1486, NIII]1750 are the same values used in
# nitrogen/compute_nitrogen_abundance.py.)
VDB = {
    # name : (flux_rel, flux_err_rel) on the same arbitrary scale as CIV
    'NV':   (2.461,  0.189),
    'NIV':  (0.258,  0.027),
    'NIII': (0.382,  0.021),
    'CIV':  (25.29,  0.11),
    'CIII': (15.49,  0.08),    # Van den Berk 2001 CIII] 1908
    'CII':  (1.787,  0.045),   # Van den Berk 2001 CII] 2326
    'MgII': (32.28,  0.15),    # Van den Berk 2001 MgII 2798 (broad)
}


def observed_ratio(num_key, denom_key='CIV'):
    """Return (ratio, err) for VDB num/denom with quadrature error propagation."""
    n, ne = VDB[num_key]
    d, de = VDB[denom_key]
    r = n / d
    re_ = r * np.sqrt((ne / n) ** 2 + (de / d) ** 2)
    return r, re_


def parse_line_list(filename):
    """Parse a Cloudy 'save line list' grid output.

    Returns:
        col_names : list[str]  (length N_lines, in order)
        rows      : np.ndarray (N_grid, N_lines)  intensities per grid point
    """
    rows = []
    col_names = None
    current_row = None
    with open(filename) as f:
        for line in f:
            if line.startswith('#lineslist'):
                # Header row: '#lineslist\t<label1>\t<label2>\t...'
                parts = line.rstrip('\n').split('\t')
                col_names = [p.strip().lstrip('#') for p in parts[1:]]
                continue
            if line.startswith('iteration'):
                # Last 'iteration N' before each GRID_DELIMIT is the converged values
                parts = line.rstrip('\n').split('\t')
                # parts[0] = 'iteration N', parts[1:] = values
                try:
                    current_row = [float(x) for x in parts[1:]]
                except ValueError:
                    current_row = None
                continue
            if line.startswith('###') and 'GRID_DELIMIT' in line:
                if current_row is not None:
                    rows.append(current_row)
                    current_row = None
                continue
    if current_row is not None:
        rows.append(current_row)
    return col_names, np.array(rows)


def parse_grd(filename):
    """Parse a Cloudy .grd file. Returns the parameter values in grid order."""
    vals = []
    with open(filename) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            # Last column is the grid parameter (metallicity here)
            vals.append(float(parts[-1]))
    return np.array(vals)


def load_grid():
    """Load all (aox, logZ, line_dict) records from the Cloudy grid."""
    line_files = sorted(glob.glob(os.path.join(GRID_DIR, 'aox*_Z_LineList_BLR_Fe2.txt')))
    if not line_files:
        raise FileNotFoundError(f'No line list files in {GRID_DIR}')

    records = []
    for lf in line_files:
        m = re.search(r'aox(-?\d+\.\d+)_Z_LineList', os.path.basename(lf))
        if not m:
            continue
        aox = float(m.group(1))
        grd_file = lf.replace('_LineList_BLR_Fe2.txt', '.grd')

        col_names, rows = parse_line_list(lf)
        logZ_vals = parse_grd(grd_file)

        if len(rows) != len(logZ_vals):
            print(f'  WARNING: {os.path.basename(lf)} has {len(rows)} grid blocks '
                  f'but .grd has {len(logZ_vals)} entries; truncating')
            n = min(len(rows), len(logZ_vals))
            rows = rows[:n]
            logZ_vals = logZ_vals[:n]

        # Build column-name -> index map once per file
        col_idx = {name: i for i, name in enumerate(col_names)}

        for logZ, row in zip(logZ_vals, rows):
            line_dict = {key: row[col_idx[label]] for key, label in LINE_COLS.items()
                         if label in col_idx}
            records.append({'aox': aox, 'logZ': logZ, 'lines': line_dict})

    return records


# Panel definitions
NITRO_PANELS = [
    ('NV',   r'$\rm N\,V\,\lambda1240 / C\,IV$'),
    ('NIV',  r'$\rm N\,IV]\,\lambda1486 / C\,IV$'),
    ('NIII', r'$\rm N\,III]\,\lambda1750 / C\,IV$'),
]
LOWION_PANELS = [
    ('MgII', r'$\rm Mg\,II\,\lambda2798 / C\,IV$'),
    ('CIII', r'$\rm C\,III]\,\lambda1909 / C\,IV$'),
    ('CII',  r'$\rm C\,II]\,\lambda2326 / C\,IV$'),
]


def make_figure(records):
    if not records:
        raise RuntimeError('No grid records to plot')

    aox_arr = np.array([r['aox'] for r in records])
    logZ_arr = np.array([r['logZ'] for r in records])

    unique_aox = np.unique(aox_arr)
    unique_logZ = np.unique(logZ_arr)

    # Find Z value closest to LOG_Z_REF
    iref = int(np.argmin(np.abs(unique_logZ - LOG_Z_REF)))
    logZ_ref = unique_logZ[iref]
    print(f'Highlighting log Z/Zsun = {logZ_ref:.2f} '
          f'(closest grid value to {LOG_Z_REF:.2f})')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    cmap = plt.cm.viridis
    norm = Normalize(vmin=unique_logZ.min(), vmax=unique_logZ.max())

    panels = NITRO_PANELS + LOWION_PANELS

    for ax, (key, ylabel) in zip(axes.flat, panels):
        # Background curves: every Z value, faint
        for logZ in unique_logZ:
            mask = logZ_arr == logZ
            sel = [r for r in records if r['logZ'] == logZ]
            x = np.array([r['aox'] for r in sel])
            y = np.array([
                r['lines'][key] / r['lines']['CIV']
                if r['lines'].get('CIV', 0) > 0 else np.nan
                for r in sel
            ])
            srt = np.argsort(x)
            is_ref = np.isclose(logZ, logZ_ref)
            ax.plot(
                x[srt], y[srt], '-',
                lw=3.5 if is_ref else 1.2,
                color=cmap(norm(logZ)),
                alpha=1.0 if is_ref else 0.35,
                zorder=3 if is_ref else 1,
            )

        # Observed BLR band
        try:
            r_obs, r_err = observed_ratio(key, 'CIV')
            ax.axhspan(r_obs - r_err, r_obs + r_err, color='red', alpha=0.18,
                       zorder=0)
            ax.axhline(r_obs, color='red', ls='--', lw=1.5, zorder=2,
                       label=r'VdB01 obs.')
        except KeyError:
            pass

        ax.set_ylabel(ylabel)
        ax.set_yscale('log')
        ax.minorticks_on()
        ax.tick_params(top=True, right=True, axis='both', which='major',
                       length=7, width=1.4, direction='in')
        ax.tick_params(top=True, right=True, axis='both', which='minor',
                       length=4, width=1, direction='in')
        ax.grid(True, which='both', alpha=0.15, lw=0.6)

    for ax in axes[1, :]:
        ax.set_xlabel(r'$\alpha_{\rm ox}$')

    # Single legend in upper-left subplot
    axes[0, 0].legend(loc='best', frameon=False, fontsize=12)

    # Single colorbar on the right
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.subplots_adjust(left=0.06, right=0.92, top=0.93, bottom=0.08,
                        hspace=0.12, wspace=0.22)
    cax = fig.add_axes([0.935, 0.10, 0.012, 0.82])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r'$\log(Z/Z_\odot)$')
    cbar.ax.tick_params(direction='in', which='both')

    fig.suptitle(
        r'Top: nitrogen test --- Bottom: low-ionization rebuttal '
        r'(thick line: $\log Z/Z_\odot = ' + f'{logZ_ref:.1f}$)',
        y=0.985, fontsize=15,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT_PATH}')
    plt.close(fig)


def main():
    records = load_grid()
    print(f'Loaded {len(records)} grid points '
          f'({len(np.unique([r["aox"] for r in records]))} alpha_ox '
          f'x {len(np.unique([r["logZ"] for r in records]))} log Z)')
    make_figure(records)


if __name__ == '__main__':
    main()
