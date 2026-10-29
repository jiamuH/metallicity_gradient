#!/usr/bin/env python3
"""LOC-averaged nitrogen abundance inference (N23 / N25 / m3 grids).

Companion to compute_nitrogen_abundance.py, which infers the nitrogen abundance
from SINGLE-ZONE models at (log n_H, log phi) = (10, 19).  Here the same
inference is done with a genuine LOC average over the (phi, n_H) grid, using the
canonical Baldwin-Korista weights Gamma = beta = -1:

    <F>_n(phi) = mean over log n of F(phi, n)              (beta = -1)
    L_line     = sum over log phi of <F>_n(phi) * phi^-1   (Gamma = -1, r ~ phi^-1/2)

Line ratios are formed from the LOC-integrated luminosities, so any common
normalisation (e.g. H-beta) cancels.

Matter-bounded clouds are excluded.  A slab of column N_H contains its own
Stromgren depth only while phi / (alpha_B n) < N_H, i.e.

    log phi - log n < log(alpha_B * N_H),

which for alpha_B = 2.6e-13 cm^3/s is 10.41 (N_H = 1e23) and 12.41 (N_H = 1e25).
Beyond that the ionization front runs out the back of the cloud and the model
line fluxes are not meaningful for the abundance inference; at N_H = 1e23 this
removes a substantial part of the high-ionization corner.

The grid variable X (`element nitrogen scale factor`) multiplies on top of
`metals`, so X = (N/alpha)/(N/alpha)_sun and (N/H)/(N/H)_sun = X * Z_alpha.

Results are written to nitrogen/loc_nitrogen_abundance.dat, leaving the existing
single-zone nitrogen_abundance_vs_r.dat untouched.
"""
from pathlib import Path

import numpy as np

from plot_aox_line_ratios import parse_line_list

ALPHA_B = 2.6e-13          # cm^3 s^-1, case-B recombination at BLR temperatures

MODEL_DIR = Path('/Users/jiamuh/c23.01/my_models/loc_metal_nitrogen')

# name: (line-list file, .grd file, Z_alpha, log10 N_H, nitrogen varied?)
GRIDS = {
    'N23_m1': ('strong_LOC_vary_nitrogen_N23_LineList_BLR_Fe2_flux.txt',
               'strong_LOC_vary_nitrogen_N23.grd', 1.0, 23.0, True),
    'N25_m1': ('strong_LOC_vary_nitrogen_N25_LineList_BLR_Fe2_flux.txt',
               'strong_LOC_vary_nitrogen_N25.grd', 1.0, 25.0, True),
    'N23_m3': ('strong_LOC_m3_N23_LineList_BLR_Fe2_flux.txt',
               'strong_LOC_m3_N23.grd', 3.0, 23.0, False),
}
# Robustness grids (referee comment 6): copies of N23_m3 with one change each.
for _name, _prefix in (('N23_m3_turb0',   'strong_LOC_m3_N23_turb0'),     # no microturbulence
                       ('N23_m3_turb300', 'strong_LOC_m3_N23_turb300'),   # 300 km/s
                       ('N23_m3_turb300diss', 'strong_LOC_m3_N23_turb300diss'),  # 300 km/s + dissipation heating
                       ('N23_m3_aox1.40', 'AGN_aoxm1.40_LOC_m3_N23'),     # Cloudy AGN SED, a_ox = -1.40
                       ('N23_m3_aox1.65', 'AGN_aoxm1.65_LOC_m3_N23'),
                       ('N23_m3_aox1.90', 'AGN_aoxm1.90_LOC_m3_N23')):
    GRIDS[_name] = (f'{_prefix}_LineList_BLR_Fe2_flux.txt', f'{_prefix}.grd', 3.0, 23.0, False)


def radii_chunks(name):
    """The (file, log phi axis, log n axis) chunk list of grid `name`, in the
    layout loc_emission_weighted_r expects, for consistent_loc_radii(chunks=...)."""
    return [(GRIDS[name][0], np.arange(17.0, 23.5, 0.5), np.arange(9.0, 12.5, 0.5))]

COLS = {'N5': 'blnd 1240.00A', 'N4': 'blnd 1486.00A', 'N3': 'blnd 1750.00A',
        'C4': 'blnd 1549.00A', 'O3': 'blnd 1666.00A'}
IONS = ('N5', 'N4', 'N3')

# Vanden Berk et al. (2001) do not detect N IV] 1486; their 'N IV' entry is
# N IV 1718, so no N4 value is listed.
VDB = {'N5': (2.461, 0.189), 'N3': (0.382, 0.021),
       'C4': (25.29, 0.11), 'O3': (0.480, 0.028)}
CSH = {'N5': (13.0, 1.5), 'N4': (0.7, 0.3), 'N3': (2.0, 0.2),
       'C4': (44.8, 1.7), 'O3': (4.1, 1.1)}


def consistent_loc_radii(beta=-1.0, gamma=-1.0, r_nv_ld=2.0, chunks=None):
    """LOC emission-weighted radii and 25-75% emission ranges for the N ions,
    computed on the SAME cloud ensemble as the abundance inference: log phi in
    PHI_COMMON, log n >= 9, all Cloudy grid cells kept (no matter-bounded cut).
    N V is anchored to r_nv_ld (NGC 5548).  The emissivities come from the
    metals-3 N23 grid, the same Z_alpha = 3 model as the abundance inference;
    `chunks` (see radii_chunks) takes them from another grid instead.
    Returns {ion: (r_ew, r25, r75)} in light-days."""
    import loc_emission_weighted_r as L
    PHI, LOGN = L.PHI_AX_FULL, L.LOGN_AX_FULL
    em = L._load_line_intensity(chunks or radii_chunks('N23_m3'))
    pp, nn = np.meshgrid(PHI, LOGN, indexing='ij')
    # Baseline ensemble: common window only, no matter-bounded cut (the
    # truncated-column clouds are genuine Cloudy predictions; see load_grid).
    mask = (nn >= 9.0) & (pp >= PHI_COMMON[0]) & (pp <= PHI_COMMON[1])
    w_n = (10.0 ** LOGN) ** (beta + 1)

    em_loc = {}
    for ln in L.LINE_INFO:
        prof = np.zeros(len(PHI))
        for ip in range(len(PHI)):
            keep = mask[ip]
            if keep.any():
                prof[ip] = np.sum(em[ln][ip, keep] * w_n[keep]) / np.sum(w_n[keep])
        em_loc[ln] = prof

    logphi_dense = np.linspace(PHI.min(), PHI.max(), 400)
    r_dense = 10 ** ((54.0 - np.log10(4 * np.pi) - logphi_dense) / 2.0)
    raw = {}
    for ln in L.LINE_INFO:
        em_d = np.interp(logphi_dense, PHI, em_loc[ln])
        f = np.maximum(em_d, 0.0) * r_dense ** (gamma + 2)
        order = np.argsort(r_dense)
        r_s, f_s = r_dense[order], f[order]
        L_tot = np.trapz(f_s, r_s)
        r_ew = 10 ** (np.trapz(np.log10(r_s) * f_s, r_s) / L_tot)
        cum = np.concatenate([[0.0],
                              np.cumsum(0.5 * (f_s[:-1] + f_s[1:]) * np.diff(r_s))])
        cum /= cum[-1]
        raw[ln] = (r_ew, np.interp(0.25, cum, r_s), np.interp(0.75, cum, r_s))
    scale = (r_nv_ld / L.LD_PER_CM) / raw['N5'][0]
    return {ln: tuple(v * scale * L.LD_PER_CM for v in t)
            for ln, t in raw.items()}


def read_grd(path):
    """Return arrays of the varied grid parameters, one row per grid point.

    Columns after the fixed header block are the varied parameters in the order
    they appear in the input file (phi, hden, and optionally the N scale factor).
    """
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            p = line.rstrip('\n').split('\t')
            rows.append([float(v) for v in p[6:-1]])
    return np.array(rows)


PHI_COMMON = (17.0, 21.0)   # phi range shared by all grids, for fair comparison


def load_grid(name, phi_range=PHI_COMMON, strom_offset=np.inf):
    """Return (params, {line: flux array}) on the common (phi, n) window.

    Baseline: NO matter-bounded cut.  The high-phi/low-n cells are genuine
    Cloudy predictions for clouds truncated at the stopping column (the
    ionization front exits the back and the low-ionization lines are weak,
    self-consistently), so the standard LOC integral keeps them.  Pass a finite
    `strom_offset` (dex, shifting the threshold log phi - log n <
    log(alpha_B N_H) + offset) to explore ionization-bounded-only ensembles as
    a robustness variant; 0.0 is the plain Stromgren criterion."""
    fname, grd, z_alpha, log_nh, varied = GRIDS[name]
    names, rows = parse_line_list(MODEL_DIR / fname)
    params = read_grd(MODEL_DIR / grd)
    if len(params) != len(rows):
        raise RuntimeError(f'{name}: {len(params)} grid rows vs {len(rows)} data rows')

    logphi, logn = params[:, 0], params[:, 1]
    x = params[:, 2] if varied else np.ones(len(params))

    bounded = (logphi - logn) < np.log10(ALPHA_B * 10 ** log_nh) + strom_offset
    in_phi = (logphi >= phi_range[0]) & (logphi <= phi_range[1])
    keep = bounded & in_phi
    print(f'  {name}: {len(rows)} points, {int((~bounded).sum())} matter-bounded, '
          f'{int((bounded & ~in_phi).sum())} outside log phi '
          f'{phi_range[0]:.0f}-{phi_range[1]:.0f}, {int(keep.sum())} kept')

    idx = {k: names.index(v) for k, v in COLS.items() if v in names}
    flux = {k: rows[keep, i] for k, i in idx.items()}
    return dict(logphi=logphi[keep], logn=logn[keep], x=x[keep],
                z_alpha=z_alpha, varied=varied), flux


def loc_integrate(logphi, logn, values):
    """LOC-integrated luminosity with Gamma = beta = -1.

    beta = -1 makes the density weight flat in log n (simple mean over the
    sampled densities at each phi); Gamma = -1 gives a radial weight
    r^(Gamma+3) = r^2 proportional to phi^-1.
    """
    total = 0.0
    for lp in np.unique(logphi):
        m = logphi == lp
        if not m.any():
            continue
        total += np.mean(values[m]) * 10 ** (-lp)
    return total


def model_ratio_curve(params, flux, num, den):
    """LOC line ratio as a function of the nitrogen scale factor."""
    xs = np.unique(params['x'])
    out = []
    for xv in xs:
        m = params['x'] == xv
        ln = loc_integrate(params['logphi'][m], params['logn'][m], flux[num][m])
        ld = loc_integrate(params['logphi'][m], params['logn'][m], flux[den][m])
        out.append(ln / ld)
    return xs, np.array(out)


def obs_ratio(src, num, den):
    n, ne = src[num]
    d, de = src[den]
    r = n / d
    return r, r * np.hypot(ne / n, de / d)


def invert(xs, ys, target):
    """(X, extrapolated_flag) where the model ratio curve equals target.

    Targets beyond the grid are converted with a power-law extrapolation using
    the local log-log slope of the top (or bottom) ~0.5 dex of the curve, and
    flagged True.  Extrapolated values sit in the regime where the curve is
    progressively saturating (nitrogen no longer a trace coolant), so they
    carry an unquantified model systematic on top of the returned value."""
    order = np.argsort(ys)
    ys_s, xs_s = ys[order], xs[order]
    if target > ys_s[-1]:
        sel = xs_s >= xs_s[-1] * 10 ** -0.5
        s = np.polyfit(np.log10(xs_s[sel]), np.log10(ys_s[sel]), 1)[0]
        return float(xs_s[-1] * (target / ys_s[-1]) ** (1.0 / s)), True
    if target < ys_s[0]:
        sel = xs_s <= xs_s[0] * 10 ** 0.5
        s = np.polyfit(np.log10(xs_s[sel]), np.log10(ys_s[sel]), 1)[0]
        return float(xs_s[0] * (target / ys_s[0]) ** (1.0 / s)), True
    return float(np.interp(target, ys_s, xs_s)), False


def main():
    print('Loading LOC grids:')
    loaded = {}
    for name in GRIDS:
        try:
            loaded[name] = load_grid(name)
        except Exception as exc:                       # noqa: BLE001
            print(f'  {name}: SKIPPED ({exc})')

    out_lines = ['# LOC-averaged nitrogen abundance inference',
                 '# X = (N/alpha)/(N/alpha)_sun ; N/H rel solar = X * Z_alpha',
                 '# Gamma = beta = -1 ; full grid on the common window (no cut)',
                 '# grid  denominator  source  ion  X  N_H_rel_solar  flag']

    for name, (params, flux) in loaded.items():
        z = params['z_alpha']
        print(f'\n=== {name}  (Z_alpha = {z:.0f}) ===')
        for den, src, sname in (('C4', VDB, 'VdB01'), ('C4', CSH, 'C&S03'),
                                ('O3', CSH, 'C&S03')):
            if den not in flux:
                continue
            print(f'  -- {den} / {sname} --')
            for ion in IONS:
                if ion not in src or den not in src:
                    continue
                r_obs, _ = obs_ratio(src, ion, den)
                xs, ys = model_ratio_curve(params, flux, ion, den)
                r_solar = float(np.interp(1.0, xs, ys)) if len(xs) > 1 else float(ys[0])
                if params['varied']:
                    xv, off = invert(xs, ys, r_obs)
                    method = 'limit' if off else 'invert'
                else:
                    # nitrogen fixed: linear scaling from the solar-N model
                    xv, off = r_obs / r_solar, False
                    method = 'scaled'
                print(f'    {ion}: model(X=1) = {r_solar:.4f}  obs = {r_obs:.4f}'
                      f'  ->  X = {xv:.2f}  (N/H = {xv * z:.2f}) [{method}]')
                out_lines.append(f'{name}  {den}  {sname}  {ion}  {xv:.4f}  '
                                 f'{xv * z:.4f}  {method}')

    out = Path('nitrogen/loc_nitrogen_abundance.dat')
    out.write_text('\n'.join(out_lines) + '\n')
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
