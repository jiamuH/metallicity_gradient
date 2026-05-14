#!/usr/bin/env python3
"""
Generate the 3 Zsun (overall metallicity) Nagao SED reference Cloudy run,
matched to nitrogen/generate_nagao_ref.py except metals = 3 instead of 1.

Same setup otherwise: hden 10, phi(H) 19, varying nitrogen scale factor
(linear grid 0.1..9.9 step 0.2 -> 50 blocks), turbulence 100 km/s, stop
column 24, no iterate convergence.

Output: /Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100_m3/
        strong_n10_phi19_N24_m3_v100.in

Run:
    python3 nitrogen/generate_nagao_ref_m3.py
then on the Cloudy side:
    cd /Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100_m3 && /Users/jiamuh/c23.01/source/cloudy.exe -r strong_n10_phi19_N24_m3_v100
"""

import os

OUTDIR = "/Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100_m3"
PREFIX = "strong_n10_phi19_N24_m3_v100"

CONTENT = f"""\
title BLR Nagao strong bump (matched to aox grid: N24, vturb 100, metals=3)
set save prefix "{PREFIX}"
init "c84.ini"
interpolate (-8.0 -13.6) (-1.7 -1.0) (-1.3 1.48) (-1.0 2.0)
continue    (-0.19 1.65) (0.4 1.32) (1.87  -1.87)  (2.43 -2.37)
continue    (2.98 -2.79) (3.3 -2.96) (3.7 -3.38) (4.18 -4.05)
continue    (4.99 -5.59) (5.1 -6.67) (7.0 -12.37)
phi(H) 19
hden 10
metals 3
element nitrogen scale factor 1 vary
grid 0.1 10 0.2 linear
turbulence 100 km/s
stop column density 24
stop temperature 3e3 K
stop zone 800
print lines column
print line faint -3
print last
save overview "_blr.ovr" last
save line list "_LineList_BLR_Fe2.txt" "LineList_BLR_Fe2.dat" last
save continuum "_SED.conA" last units Angstroms
save grid "_.grd"
"""


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, PREFIX + ".in")
    with open(path, "w") as fh:
        fh.write(CONTENT)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
