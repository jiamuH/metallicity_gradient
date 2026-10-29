"""SDSS-RM per-object mean +/- error-of-the-mean of log10(ratio) for
MgII/CIV and (SiIV+OIV)/CIV -> new OBS_RANGE values.  Scratch query."""
import glob
import os
import numpy as np
from alpha.loc_gradient_fit import load_object_data, _OBS_DATA_DIR

files = sorted(glob.glob(os.path.join(_OBS_DATA_DIR, '*_line_ratios.dat')))
for t, key in [('mg2_c4', 'MgII/CIV'), ('si4_c4', '(SiIV+OIV)/CIV')]:
    x = []
    for f in files:
        rm_id = os.path.basename(f).split('_')[0]
        try:
            data, _ = load_object_data(rm_id)
        except Exception:
            continue
        if t not in data:
            continue
        R = np.asarray(data[t]['R'], float)
        R = R[np.isfinite(R) & (R > 0)]
        if R.size:
            x.append(np.mean(R))                 # object ratio = mean over its epochs
    x = np.log10(np.array(x))
    N = x.size
    mean = float(np.mean(x))
    sem = float(np.std(x, ddof=1) / np.sqrt(N))
    print(f"{key:18s}  N={N:4d}  mean(log)={mean:+.4f}  SEM={sem:.4f} dex"
          f"   ->  OBS_RANGE: ({mean:+.4f}, {sem:.4f})")
