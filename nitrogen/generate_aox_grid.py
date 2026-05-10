#!/usr/bin/env python3
"""
Generate Cloudy .in files for an alpha_ox x metallicity grid that saves the
full BLR line list (so NV, NIV], NIII], CIV, MgII, CIII], CII], etc. are all
available for the SED-vs-N-abundance comparison plot).

Mirrors /Users/jiamuh/c23.01/my_models/j0601_grid/generate_grid.py but:
  - writes .in files to /Users/jiamuh/c23.01/my_models/aox_nitrogen_grid/
  - adds the 'save line list' directive so all ~500 BLR lines are saved
    in one tab-separated file per alpha_ox value.

Run:
    python3 nitrogen/generate_aox_grid.py
then on the Cloudy side:
    cd /Users/jiamuh/c23.01/my_models/aox_nitrogen_grid && ./run_all.sh
"""

import os
import numpy as np

OUTDIR = "/Users/jiamuh/c23.01/my_models/aox_nitrogen_grid"
PREFIX_FMT = "aox{aox:.1f}_Z"

TEMPLATE = """\
title aox={aox:.1f} metallicity grid (j0601-style for nitrogen comparison)
set save prefix "aox{aox:.1f}_Z"
AGN 5.0 {aox:.1f} -0.5 -1.0
hden 9
ionization parameter -1.0
metals -1 vary
grid -2 1 0.3
turbulence 100 km/s
stop column density 24
stop temperature 3000 K
stop zone 800
iterate convergence
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


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    run_lines = []
    for aox in aox_vals:
        prefix = PREFIX_FMT.format(aox=aox)
        fname = f"{prefix}.in"
        with open(os.path.join(OUTDIR, fname), "w") as f:
            f.write(TEMPLATE.format(aox=aox))
        run_lines.append(prefix)

    run_sh = os.path.join(OUTDIR, "run_all.sh")
    with open(run_sh, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {OUTDIR}\n")
        f.write('CLOUDY_BIN="../../source/cloudy.exe"\n\n')
        for prefix in run_lines:
            f.write(f'echo "Running {prefix}" && "$CLOUDY_BIN" -r {prefix}\n')
    os.chmod(run_sh, 0o755)

    print(f"Generated {len(run_lines)} .in files and run_all.sh in {OUTDIR}")


if __name__ == "__main__":
    main()
