"""Average observed MgII/CIV over the SDSS-RM sample.  Scratch query."""
import glob
import os
import numpy as np
from alpha.loc_gradient_fit import load_object_data, _OBS_DATA_DIR

files = sorted(glob.glob(os.path.join(_OBS_DATA_DIR, '*_line_ratios.dat')))
per_obj, all_ep = [], []
n_with_mg = 0
for f in files:
    rm_id = os.path.basename(f).split('_')[0]
    try:
        data, _ = load_object_data(rm_id)
    except Exception as e:
        print(f"  skip {rm_id}: {e}")
        continue
    if 'mg2_c4' not in data:
        continue
    R = np.asarray(data['mg2_c4']['R'], float)
    R = R[np.isfinite(R) & (R > 0)]
    if R.size == 0:
        continue
    n_with_mg += 1
    per_obj.append(np.mean(R))
    all_ep.extend(R.tolist())

per_obj = np.array(per_obj)
all_ep = np.array(all_ep)
print(f"\nfiles: {len(files)};  objects with MgII/CIV: {n_with_mg};  total epochs: {all_ep.size}")
for label, x in [('per-epoch (all epochs pooled)', all_ep),
                  ('per-object (each object = its epoch mean)', per_obj)]:
    print(f"\n{label}:")
    print(f"  N            = {x.size}")
    print(f"  mean  ratio  = {np.mean(x):.3f}   log = {np.log10(np.mean(x)):+.3f}")
    print(f"  median ratio = {np.median(x):.3f}   log = {np.log10(np.median(x)):+.3f}")
    print(f"  mean of log  = {np.mean(np.log10(x)):+.3f}  +/- {np.std(np.log10(x)):.3f} dex")
    print(f"  range        = {x.min():.3f} .. {x.max():.3f}")
