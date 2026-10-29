"""Eddington ratio and mass of the Constantin & Shields NLS1s with the
UV-extrapolated L(5100) for every object, against the version that uses the
published L(5100) for the UV-absorbed objects.

Run: ~/miniconda3/envs/pypeit/bin/python -m nitrogen._compare_l5100_routes
"""
import numpy as np
from nitrogen.csh_nls1_properties import compute


def main():
    uv, pub = compute(use_published=False), compute(use_published=True)
    print(f"{'object':17s} {'logM uv':>8s} {'logM pub':>8s} {'logEdd uv':>9s} {'logEdd pub':>10s} {'diff':>6s}")
    for a, b in zip(uv, pub):
        if a['log_edd'] == b['log_edd']:
            continue
        print(f"{a['name']:17s} {a['log_m']:8.2f} {b['log_m']:8.2f} {a['log_edd']:9.2f} {b['log_edd']:10.2f}"
              f" {b['log_edd'] - a['log_edd']:6.2f}")
    for lab, rows in (('UV only', uv), ('published', pub)):
        e = np.array([r['log_edd'] for r in rows])
        m = np.array([r['log_m'] for r in rows])
        print(f'{lab:10s} logEdd median {np.median(e):5.2f} range {e.min():5.2f} to {e.max():5.2f};'
              f' logM median {np.median(m):5.2f}')


if __name__ == '__main__':
    main()
