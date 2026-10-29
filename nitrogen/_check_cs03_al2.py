"""Is the Constantin & Shields (2003) Al II 1670 component too strong?
Rebuild their Table 3 Lorentzians from C IV to Si II 1817 (including the broad
C IV and He II components) on the recovered composite, fit a straight-line
continuum, then refit the O III] and Al II equivalent widths with every other
line held at the published values.

Run (from the project root): python3 nitrogen/_check_cs03_al2.py
"""
import numpy as np
from scipy.optimize import least_squares

from plot_cs03_n3_region import lorentz, DATA

# label, centre [A], FWHM [km/s], EW [A]   (Constantin & Shields 2003 Table 3)
TABLE = [
    ('C IV narrow',  1549.5, 1900, 57.3),
    ('C IV broad',   1541.6, 12000, 19.8),
    ('He II narrow', 1639.7, 1780, 11.4),
    ('He II broad',  1630.4, 11990, 30.1),
    ('O III]',       1665.4, 2500, 6.2),
    ('Al II',        1680.0, 2250, 1.9),
    ('N IV+Al II',   1725.6, 4710, 3.3),
    ('N III]',       1750.5, 1480, 3.0),
    ('Fe II UV191',  1785.3, 1430, 1.8),
    ('Si II',        1816.9, 1250, 0.9),
]
FREE = ('O III]', 'Al II')
WMIN, WMAX = 1600.0, 1840.0


def model(w, ews, cont_ab):
    shape = 1.0 + sum(lorentz(w, c, fw, ews[lab]) for lab, c, fw, _ in TABLE)
    return (cont_ab[0] * w + cont_ab[1]) * shape, shape


def main():
    w, f = np.loadtxt(DATA, unpack=True)
    s = (w > WMIN) & (w < WMAX)
    w, f = w[s], f[s]
    ews = {lab: ew for lab, _, _, ew in TABLE}

    _, shape = model(w, ews, (0, 1))
    ab = np.polyfit(w, f / shape, 1)
    m, _ = model(w, ews, ab)
    win = (w > 1655) & (w < 1700)
    print(f'published fit: rms(all) {np.std(f - m):.3f}, mean residual 1655-1700 A {np.mean((f - m)[win]):+.3f}')

    def resid(p):
        e = dict(ews, **{'O III]': p[0], 'Al II': p[1]})
        mm, _ = model(w, e, p[2:])
        return f - mm
    sol = least_squares(resid, [6.2, 1.9, *ab], bounds=([0, 0, -1, -10], [30, 30, 1, 10]))
    J = sol.jac
    cov = np.linalg.inv(J.T @ J) * np.sum(sol.fun ** 2) / (len(w) - len(sol.x))
    err = np.sqrt(np.diag(cov))
    print(f'refit: EW(O III]) = {sol.x[0]:.2f} +- {err[0]:.2f} A (published 6.2), '
          f'EW(Al II) = {sol.x[1]:.2f} +- {err[1]:.2f} A (published 1.9)')
    m2 = f - sol.fun
    print(f'refit: rms(all) {np.std(sol.fun):.3f}, mean residual 1655-1700 A {np.mean(sol.fun[win]):+.3f}')


if __name__ == '__main__':
    main()


def robustness():
    """Repeat the refit for other windows and a quadratic continuum."""
    w0, f0 = np.loadtxt(DATA, unpack=True)
    ews0 = {lab: ew for lab, _, _, ew in TABLE}
    for wmin, wmax, deg in ((1600, 1840, 1), (1600, 1840, 2), (1620, 1720, 1), (1645, 1710, 1)):
        s = (w0 > wmin) & (w0 < wmax)
        w, f = w0[s], f0[s]

        def resid(p):
            e = dict(ews0, **{'O III]': p[0], 'Al II': p[1]})
            shape = 1.0 + sum(lorentz(w, c, fw, e[lab]) for lab, c, fw, _ in TABLE)
            return f - np.polyval(p[2:], w - 1700) * shape
        p0 = [6.2, 1.9] + [0] * deg + [1.0]
        sol = least_squares(resid, p0)
        print(f'  window {wmin}-{wmax} A, continuum degree {deg}: '
              f'EW(O III]) = {sol.x[0]:.2f}, EW(Al II) = {sol.x[1]:.2f}')


robustness()
