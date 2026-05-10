#!/usr/bin/env python3
"""
Generate Cloudy .in files for an alpha_ox x metallicity grid that saves the
full BLR line list (so NV, NIV], NIII], CIV, MgII, CIII], CII], etc. are all
available for the SED-vs-N-abundance comparison plot).

Mirrors /Users/jiamuh/c23.01/my_models/j0601_grid/generate_grid.py but:
  - writes .in files to /Users/jiamuh/c23.01/my_models/aox_nitrogen_grid/
  - uses hden 10, phi(H) = 19 (matches the existing Nagao SED reference at
    /Users/jiamuh/c23.01/my_models/singlezone_nitrogen_series/
    strong_n10_phi19_N23_m1_v0 so the comparison is apples-to-apples)
  - log Z grid from -1 to 1 (0.1 to 10 Zsun) to keep run-time modest
  - no 'iterate convergence' (kept off for runtime; line ratios are slightly
    less accurate but the qualitative trends with alpha_ox are unaffected)
  - adds the 'save line list' directive so all ~500 BLR lines are saved
    in one tab-separated file per alpha_ox value.

Run:
    python3 nitrogen/generate_aox_grid.py
then on the Cloudy side:
    cd /Users/jiamuh/c23.01/my_models/aox_nitrogen_grid && python3 run_all.py
"""

import os
import numpy as np

OUTDIR = "/Users/jiamuh/c23.01/my_models/aox_nitrogen_grid"
PREFIX_FMT = "aox{aox:.1f}_Z"

TEMPLATE = """\
title aox={aox:.1f} metallicity grid (j0601-style for nitrogen comparison)
set save prefix "aox{aox:.1f}_Z"
AGN 5.0 {aox:.1f} -0.5 -1.0
hden 10
phi(H) 19
metals -1 vary
grid -1 1 0.2
turbulence 100 km/s
stop column density 24
stop temperature 3000 K
stop zone 800
print lines column
print line faint -3
print last
save overview ".ovr" last
save continuum ".con" last units Angstroms
save line emissivity ".ems" last
Si 4 1396.76A
O  4 1401.16A
blnd 1549
blnd 1240
He 2 1640.43A
blnd 1909
blnd 2798
H  1 4861.32A
H  1 1215.67A
end of lines
save line list "_LineList_BLR_Fe2.txt" "LineList_BLR_Fe2.dat" last
save grid ".grd"
"""

aox_vals = np.arange(-2.0, -0.9, 0.1)  # -2.0, -1.9, ..., -1.0

RUN_ALL_PY = '''\
#!/usr/bin/env python3
"""Run all aox*.in files through Cloudy with a tqdm progress bar."""

import glob
import os
import subprocess
import sys
import time

from tqdm import tqdm

OUTDIR = os.path.dirname(os.path.abspath(__file__))
CLOUDY_BIN = os.path.abspath(os.path.join(OUTDIR, "..", "..", "source", "cloudy.exe"))

prefixes = sorted(
    os.path.basename(p)[:-3]
    for p in glob.glob(os.path.join(OUTDIR, "aox-*_Z.in"))
)

if not prefixes:
    sys.exit("No aox-*.in files found in this directory.")
if not os.path.isfile(CLOUDY_BIN):
    sys.exit(f"Cloudy binary not found at {CLOUDY_BIN}")

print(f"Running {len(prefixes)} models with {CLOUDY_BIN}")

t0 = time.time()
pbar = tqdm(prefixes, desc="Cloudy", unit="model")
for prefix in pbar:
    pbar.set_postfix_str(prefix)
    with open(os.path.join(OUTDIR, prefix + ".log"), "w") as logf:
        subprocess.run(
            [CLOUDY_BIN, "-r", prefix],
            cwd=OUTDIR,
            stdout=logf, stderr=subprocess.STDOUT,
            check=False,
        )
elapsed = time.time() - t0
print(f"Done in {elapsed/60:.1f} min")
'''


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    run_lines = []
    for aox in aox_vals:
        prefix = PREFIX_FMT.format(aox=aox)
        fname = f"{prefix}.in"
        with open(os.path.join(OUTDIR, fname), "w") as f:
            f.write(TEMPLATE.format(aox=aox))
        run_lines.append(prefix)

    run_py = os.path.join(OUTDIR, "run_all.py")
    with open(run_py, "w") as f:
        f.write(RUN_ALL_PY)
    os.chmod(run_py, 0o755)

    print(f"Generated {len(run_lines)} .in files and run_all.py in {OUTDIR}")


if __name__ == "__main__":
    main()
