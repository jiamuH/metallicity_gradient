"""Scratch test: the log U <= -3 LOC cut is active, the two integral
implementations agree, and the cut changes the answer.  Not part of the pipeline.
"""
import numpy as np
import alpha.loc_gradient_fit as M
from alpha.loc_gradient_fit import (model_line_ratios, model_ratios_at,
                                    make_interpolators, make_interpolator4,
                                    LOGU_LO_LOC)

logQ = np.array([54.0, 55.0, 56.0])
k, p, lrr, z = 0.0, 0.35, 16.7, 3.0

interps = make_interpolators()
interp4 = make_interpolator4()

rr, _, _ = model_line_ratios(interps, logQ, k, p, log_r_ref=lrr, z_norm=z)
r_mg2, r_si2 = model_ratios_at(interp4, logQ, k, p, lrr, z, n_x=80, n_n=24)

print(f"LOGU_LO_LOC = {LOGU_LO_LOC}")
print(f"logQ                       = {logQ}")
print(f"model_line_ratios MgII/CIV = {rr['MgII/CIV']}")
print(f"model_ratios_at   MgII/CIV = {r_mg2}")
print(f"model_line_ratios SiIV/CIV = {rr['(SiIV+OIV)/CIV']}")
print(f"model_ratios_at   SiIV/CIV = {r_si2}")
assert np.allclose(rr['MgII/CIV'], r_mg2, rtol=1e-6, equal_nan=True), "MgII/CIV mismatch"
assert np.allclose(rr['(SiIV+OIV)/CIV'], r_si2, rtol=1e-6, equal_nan=True), "SiIV/CIV mismatch"
print("OK: both LOC integrals agree (n_x=80, n_n=24).")

# confirm the cut bites: temporarily disable it and compare
M.LOGU_LO_LOC = -np.inf
rr_nocut, _, _ = model_line_ratios(interps, logQ, k, p, log_r_ref=lrr, z_norm=z)
M.LOGU_LO_LOC = LOGU_LO_LOC                                  # restore
print(f"\nno-cut  MgII/CIV           = {rr_nocut['MgII/CIV']}")
print(f"with-cut MgII/CIV          = {rr['MgII/CIV']}")
d = np.log10(rr['MgII/CIV']) - np.log10(rr_nocut['MgII/CIV'])
print(f"delta log(MgII/CIV)        = {d}  (cut - nocut)")
print("cut is active" if np.any(np.abs(d) > 1e-6) else "WARNING: cut had no effect here")
