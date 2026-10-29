"""BLR radii for N III] 1750, N IV] 1486, N V 1240 from the LOC line-emission
integrand, at the canonical density weighting beta = -1 and radial index gamma = -1.

For each line:
  1. density-average the absolute line flux:
       <F>_n(log phi) = sum_n F_line(phi, n) * n^{beta+1} / sum_n n^{beta+1}.
  2. build the LOC radial integrand  f(r) = <F>_n(r) * r^{gamma+2}  (= dL/dr),
     mapping phi -> r with the universal  r ~ phi^{-1/2}  scaling.
  3. r_ew = log-space emission-weighted mean radius
       10^[ int log10(r) f(r) dr / int f(r) dr ];  r_lo/r_hi = 25/75 percentiles
       of the cumulative emission int f dr.
  4. anchor: rescale so N V's r_ew = 2 light-days (NGC 5548 reverberation-mapping value).

Grid: stitched m=1, N=23, log phi in [16, 23] step 0.5 (15 values), log n_H in
      [7, 12] step 0.5 (11 values), three Cloudy runs:
        strong_LOC_m1_N23 (phi 17-23, n 9-12),
        strong_LOC_m1_N23_lown (phi 17-23, n 7-8.5),
        strong_LOC_m1_N23_lophi (phi 16-16.5, full n 7-12).

Run from repo root:
    python3 -m nitrogen.loc_emission_weighted_r
"""
import os
import numpy as np
import pandas as pd

GRID_DIR  = '/Users/jiamuh/c23.01/my_models/loc_metal_nitrogen'
# use the `_flux.txt` (Cloudy `save line list absolute`) variants -- absolute line intensities
# (4 pi J, erg / cm^2 / s).  The non-`_flux.txt` files are normalized to H-beta and so contain
# a per-cell-varying denominator that is not appropriate for LOC averaging.
CHUNKS = [
    ('strong_LOC_m1_N23_LineList_BLR_Fe2_flux.txt',       np.arange(17.0, 23.5, 0.5), np.arange(9.0, 12.5, 0.5)),
    ('strong_LOC_m1_N23_lown_LineList_BLR_Fe2_flux.txt',  np.arange(17.0, 23.5, 0.5), np.arange(7.0,  9.0, 0.5)),
    ('strong_LOC_m1_N23_lophi_LineList_BLR_Fe2_flux.txt', np.arange(16.0, 17.0, 0.5), np.arange(7.0, 12.5, 0.5)),
]
LINE_INFO = {                                                    # column name in the line-list file
    'N3': dict(col='blnd 1750.00A'),
    'N4': dict(col='blnd 1486.00A'),
    'N5': dict(col='blnd 1240.00A'),
}
BETA_N_LOC = -1.0           # canonical LOC density weight (n^beta per dn -> flat in log n)
R_NV_LD    = 2.0            # anchor: N V peak  ==  2 light-days  (NGC 5548 RM)
FWHM_DEX   = 0.5            # half-max threshold in log <EW>_n
PHI_AX_FULL  = np.arange(16.0, 23.5, 0.5)        # 15 values
LOGN_AX_FULL = np.arange(7.0, 12.5, 0.5)         # 11 values
LD_PER_CM    = 1.0 / 2.59020684e15


def _read_grid(path):
    df = pd.read_csv(path, sep='\t', header=0)
    return df[~df.iloc[:, 0].astype(str).str.contains('GRID_DELIMIT')].reset_index(drop=True)


def _load_line_intensity(chunks=None):
    """Load line intensities (rel to H-beta, as Cloudy writes them) on the stitched
    (PHI_AX_FULL, LOGN_AX_FULL) grid.  No EW / continuum normalization -- LOC averages
    must be done on the LINE FLUX itself, not on the per-cell EW (see paper note).
    `chunks` replaces CHUNKS (same (file, phi axis, n axis) layout) to use another grid."""
    nP, nN = len(PHI_AX_FULL), len(LOGN_AX_FULL)
    em = {ln: np.full((nP, nN), np.nan) for ln in LINE_INFO}
    for fname, phi_ax, n_ax in (chunks or CHUNKS):
        df = _read_grid(os.path.join(GRID_DIR, fname))
        assert len(df) == len(phi_ax) * len(n_ax), f'{fname}: {len(df)} rows != {len(phi_ax)*len(n_ax)}'
        for jp, phi in enumerate(phi_ax):
            ip = int(np.argmin(np.abs(PHI_AX_FULL - phi)))
            for in_, n in enumerate(n_ax):
                in_full = int(np.argmin(np.abs(LOGN_AX_FULL - n)))
                row = df.iloc[jp * len(n_ax) + in_]
                for ln, lc in LINE_INFO.items():
                    em[ln][ip, in_full] = float(row[lc['col']])
    for ln in LINE_INFO:
        em[ln] = np.nan_to_num(em[ln], nan=0.0)
    return em


GAMMA_LOC = -1.0            # LOC radial weight: integrand ~ <EW>_n * r^{Gamma+2}


def loc_emission_weighted_radii(beta=BETA_N_LOC, gamma=GAMMA_LOC, r_nv_ld=R_NV_LD):
    """Return {'N3','N4','N5': (r_ew, r_lo, r_hi)} in light-days.

    Full LOC integral:
      <F>_n(phi)  = sum_n F_line(phi, n) * n^{beta+1} / sum_n n^{beta+1}  (density-averaged line flux)
      f(r)        = <F>_n(r) * r^{gamma+2}                               (LOC radial integrand, dL/dr)
      r_ew        = 10^[ int log10(r) f(r) dr / int f(r) dr ]            (log-space emission-weighted mean)
      r_lo, r_hi  = 25 / 75 percentiles of the cumulative int f dr

    Q-gamma is fixed by anchoring N V's r_ew to r_nv_ld (default 2 light-days).
    """
    em = _load_line_intensity()
    n_lin = 10 ** LOGN_AX_FULL
    n_wt  = n_lin ** (beta + 1)
    n_wt /= n_wt.sum()

    # density-averaged LINE FLUX at each phi (NOT EW -- proper LOC averaging is on line flux)
    em_loc = {ln: np.sum(em[ln] * n_wt[None, :], axis=1) for ln in LINE_INFO}

    # arbitrary Q for the r-axis; final rescale anchors NV at r_nv_ld
    logphi_dense = np.linspace(PHI_AX_FULL.min(), PHI_AX_FULL.max(), 400)
    log4pi = np.log10(4 * np.pi)
    log_q_arb = 54.0
    logr_dense = (log_q_arb - log4pi - logphi_dense) / 2.0
    r_dense    = 10 ** logr_dense
    raw = {}
    for ln in LINE_INFO:
        em_d = np.interp(logphi_dense, PHI_AX_FULL, em_loc[ln])
        f    = np.maximum(em_d, 0.0) * r_dense ** (gamma + 2)            # dL/dr ∝ line emissivity * r^{gamma+2}
        order = np.argsort(r_dense)
        r_s, f_s = r_dense[order], f[order]
        L_tot = np.trapz(f_s, r_s)
        r_ew  = 10 ** (np.trapz(np.log10(r_s) * f_s, r_s) / L_tot)   # log-space emission-weighted mean
        cum   = np.concatenate([[0.0], np.cumsum(0.5 * (f_s[:-1] + f_s[1:]) * np.diff(r_s))])
        cum  /= cum[-1]
        r_lo  = np.interp(0.25, cum, r_s)            # 25/75 percentiles (narrower than 16/84)
        r_hi  = np.interp(0.75, cum, r_s)
        raw[ln] = (r_ew, r_lo, r_hi)
    scale = (r_nv_ld / LD_PER_CM) / raw['N5'][0]                          # anchor NV at r_nv_ld
    return {ln: tuple(v * scale * LD_PER_CM for v in trip) for ln, trip in raw.items()}


def main():
    r = loc_emission_weighted_radii()
    print(f'{"line":4s} {"r_ew [ld]":>14s}  {"r_lo [ld]":>12s}  {"r_hi [ld]":>12s}')
    for ln in ('N3', 'N4', 'N5'):
        rew, rlo, rhi = r[ln]
        print(f'{ln:4s} {rew:14.3f}  {rlo:12.3f}  {rhi:12.3f}')


if __name__ == '__main__':
    main()
