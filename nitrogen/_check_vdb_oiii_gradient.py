"""Debug: how the nitrogen gradient k_N and the offset delta change if O III]/C IV in the
Constantin & Shields (2003) composite is replaced by the Vanden Berk et al. (2001) value
(0.019 instead of 0.092), keeping the fractional error of the O III] flux.

Run: python3 nitrogen/_check_vdb_oiii_gradient.py
"""
import numpy as np

import loc_nitrogen_abundance as L
import joint_gradient_fit as J


def summary(tag, n_steps):
    samp, rows = J.run_fit(n_steps=n_steps)
    m, d = samp[:, 0], samp[:, 2]
    q = lambda x: np.percentile(x, [16, 50, 84])
    (m16, m50, m84), (d16, d50, d84) = q(m), q(d)
    print(f'{tag}: k_N = {m50:.2f} +{m84 - m50:.2f} -{m50 - m16:.2f}   '
          f'delta = {d50:.2f} +{d84 - d50:.2f} -{d50 - d16:.2f}   '
          f'(delta at prior edge 1.0: {np.mean(d > 0.95):.2f} of samples)')
    for r in rows:
        print(f'    {r["ion"]}/{r["den"]}: log X = {r["log_x"]:+.2f}')


if __name__ == '__main__':
    N = 80000
    summary('fiducial  (O III]/C IV = 0.092)', N)
    o3, o3err = L.CSH['O3']
    scale = 0.019 / (o3 / L.CSH['C4'][0])
    new = (o3 * scale, o3err * scale)
    for mod in (L, J):
        if hasattr(mod, 'CSH'):
            mod.CSH['O3'] = new
    summary('Vanden Berk (O III]/C IV = 0.019)', N)
