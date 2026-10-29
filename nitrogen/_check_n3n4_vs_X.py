#!/usr/bin/env python3
"""Is the model N III]/N IV] ratio independent of the nitrogen abundance?

The N III]-N IV] abundance inversion divides the observed ratio by the model
ratio, and the model ratio is evaluated at a reference nitrogen abundance
(scaled-solar, X = 1, in the varym metallicity map).  Because both lines are
nitrogen lines, a change in X should largely cancel from their ratio; the
residual is the difference between their local slopes dlogR/dlogX.

This checks the size of that residual on the nitrogen-varied LOC grids
(N23_m1, N25_m1), over the range of X the analysis actually infers.

Diagnostic only; writes nothing.
"""
import numpy as np

from loc_nitrogen_abundance import load_grid, model_ratio_curve


def main():
    for name in ('N23_m1', 'N25_m1'):
        params, flux = load_grid(name)
        xs, r34 = model_ratio_curve(params, flux, 'N3', 'N4')
        r_at = lambda x: float(np.interp(x, xs, r34))   # noqa: E731
        base = r_at(1.0)
        print(f'\n=== {name}: model N III]/N IV] vs nitrogen scale factor X ===')
        print(f'  {"X":>6} {"N3/N4":>8} {"rel. to X=1":>12}')
        for x in (0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 9.0):
            if x < xs.min() or x > xs.max():
                continue
            v = r_at(x)
            print(f'  {x:>6.1f} {v:>8.3f} {v / base:>11.2f}x')
        # spread over the inferred range of X (roughly 0.2 - 1.5 for N3/N4)
        lo, hi = r_at(0.2), r_at(1.5)
        print(f'  drift over X = 0.2-1.5: {abs(np.log10(hi / lo)):.3f} dex '
              f'({max(hi, lo) / min(hi, lo):.2f}x)')

        # individual line slopes, which drive the residual
        for ion in ('N3', 'N4'):
            _, r = model_ratio_curve(params, flux, ion, 'C4')
            s = np.polyfit(np.log10(xs), np.log10(r), 1)[0]
            print(f'  mean dlogR/dlogX for {ion}/C IV: {s:.3f}')


if __name__ == '__main__':
    main()
