#!/usr/bin/env python3
"""The canonical nitrogen result: one set of points, one fit, shared by all figures.

Defined to match plots/nitrogen/Nitrogen_abundance_vs_rBLR_loc_cs03_joint.png,
which is the reference figure:

  points : Constantin & Shields (2003), both reference lines, all three ions
           (six points).  Squares are ratios to C IV, triangles to O III] 1663.
           Pale bars show the Z_alpha = 1-10 Zsun systematic.
  fit    : the JOINT fit (joint_gradient_fit.py) -- one slope k_N shared by both
           reference lines, plus an offset delta between them, since the C IV
           points measure N/C and the O III] points N/O.  The N V / O III] point
           is extrapolated beyond the model grid and is excluded from the fit.

X == (N/alpha)/(N/alpha)_sun is what a line ratio constrains; N/H = X * Z_alpha
with Z_alpha = 3 from the alpha-element analysis.

Any figure showing nitrogen points or the nitrogen gradient should import from
here rather than re-deriving either.
"""
import numpy as np

from loc_nitrogen_abundance import CSH, consistent_loc_radii, obs_ratio, invert
from plot_loc_nitrogen_composites import infer
import solve_N_and_Z as S
from loc_nitrogen_abundance import load_grid

IONS = ('N5', 'N4', 'N3')
Z_ADOPT = 3.0

# (legend label, composite, reference line, marker, x-offset in dex)
DATASETS = [
    (r'$\rm CS03\,/\,C\,IV$',   CSH, 'C4', 's',  0.000),
    (r'$\rm CS03\,/\,O\,III]$', CSH, 'O3', '^',  0.045),
]
FIT_DEN = 'C4'            # the scale the fit's intercept refers to

ION_COLOR = {'N3': '#5B2B7B', 'N4': '#C2185B', 'N5': '#E67E22'}
ION_LABEL = {'N3': r'$\rm N\,III]$', 'N4': r'$\rm N\,IV]$', 'N5': r'$\rm N\,V$'}
ION_POT = {'N3': 47.4, 'N4': 77.5, 'N5': 97.9}
Z_SYS = (1.0, 10.0)

_GRIDS_READY = False


def _ensure_grids():
    global _GRIDS_READY
    if not _GRIDS_READY:
        S.GRID_X = load_grid('N25_m1')
        S.GRID_Z = S.load_varym()
        _GRIDS_READY = True


def points(z_alpha=Z_ADOPT, radii=None):
    """{(ion, den): dict(r, r_lo, r_hi, x, x_lo, x_hi, nh, nh_lo, nh_hi,
                         sys_lo, sys_hi, off)}.

    sys_lo/sys_hi bracket the plotted N/alpha when the models assume Z_alpha = 1
    or 10 Zsun instead of the adopted value.  As in the reference figure, the
    separable (X, Z) model supplies only the RELATIVE change,

        sys = x * X_sep(Z_sys) / X_sep(Z_adopt),

    so its absolute offset from the metals-3 inference cancels.  They exclude the
    Z_alpha factor that converts N/alpha to N/H, because on an N/H axis that
    factor is already carried by the measured alpha-element profile."""
    _ensure_grids()
    radii = radii or consistent_loc_radii()
    gx = np.logspace(np.log10(0.05), np.log10(9.0), 400)
    out = {}
    for _, src, den, _, _ in DATASETS:
        for ion in IONS:
            x, lo, hi, off = infer(ion, src, den)
            r, r_lo, r_hi = radii[ion]
            r_o, _ = obs_ratio(src, ion, den)
            m_ref, _ = S.build_model(ion, den, gx, np.full_like(gx, z_alpha))
            x_ref = invert(gx, m_ref, r_o)[0]
            sysv = []
            for zs in Z_SYS:
                m_alt, _ = S.build_model(ion, den, gx, np.full_like(gx, zs))
                sysv.append(x * invert(gx, m_alt, r_o)[0] / x_ref)
            out[(ion, den)] = dict(r=r, r_lo=r_lo, r_hi=r_hi,
                                   x=x, x_lo=lo, x_hi=hi,
                                   nh=x * z_alpha, nh_lo=lo * z_alpha,
                                   nh_hi=hi * z_alpha,
                                   sys_lo=min(sysv), sys_hi=max(sysv), off=off)
    return out


def fit(radii=None):
    """The joint fit: samples[:, (k_N, b, delta, ln s)] and a summary dict.
    `radii` replaces the emission radii used by the fit (default: canonical)."""
    import joint_gradient_fit as J
    canonical = J.RADII
    if radii:
        J.RADII = radii
    try:
        samp, _ = J.run_fit()
    finally:
        J.RADII = canonical
    k_lo, k_med, k_hi = np.percentile(samp[:, 0], [16, 50, 84])
    d_lo, d_med, d_hi = np.percentile(samp[:, 2], [16, 50, 84])
    return samp, dict(k=k_med, k_lo=k_lo, k_hi=k_hi,
                      delta=d_med, delta_lo=d_lo, delta_hi=d_hi)


def fit_range(pts, n=300):
    """Radii over which the fit band is drawn, matching the reference figure:
    from N V's inner emission edge to N III]'s outer edge, padded 0.1 dex."""
    return np.logspace(np.log10(pts[('N5', FIT_DEN)]['r_lo']) - 0.1,
                       np.log10(pts[('N3', FIT_DEN)]['r_hi']) + 0.1, n)


def summary_line(s):
    """The two-line annotation used on the reference figure."""
    return (r'$k_{\rm N} \equiv \dfrac{d\log\,({\rm N/H})}{d\log r} = '
            r'%.2f^{+%.2f}_{-%.2f}$' % (s['k'], s['k_hi'] - s['k'],
                                        s['k'] - s['k_lo'])
            + '\n'
            + r'$\delta = %.2f^{+%.2f}_{-%.2f}\,{\rm dex}$'
            % (s['delta'], s['delta_hi'] - s['delta'],
               s['delta'] - s['delta_lo']))


if __name__ == '__main__':
    p = points()
    _, s = fit()
    print(f'Canonical nitrogen result (Z_alpha = {Z_ADOPT:.0f}):')
    for (ion, den), d in p.items():
        print(f'  {ion}/{den}: r = {d["r"]:5.2f} ld   N/alpha = {d["x"]:6.2f}   '
              f'N/H = {d["nh"]:6.2f}' + ('  (extrapolated)' if d['off'] else ''))
    print(f'  k_N = {s["k"]:.2f} (+{s["k_hi"] - s["k"]:.2f}, '
          f'-{s["k"] - s["k_lo"]:.2f})   delta = {s["delta"]:.2f}')
