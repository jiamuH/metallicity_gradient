#!/usr/bin/env python3
"""
Generate a refreshed Nagao SED reference Cloudy run, matched to the alpha_ox
grid setup so the comparison line in plot_aox_line_ratios.py is truly
apples-to-apples (only the SED differs).

Mirrors /Users/jiamuh/c23.01/my_models/singlezone_nitrogen_series/
strong_m1_n10_phi_19_N23_v0.in (Nagao "strong bump" interpolated continuum,
hden 10, phi(H) 19, solar metals, varying nitrogen scale factor) but adds:
  - stop column density 24  (was 23)
  - turbulence 100 km/s     (was commented out)
  (no iterate convergence: matches the aox grid for apples-to-apples
   comparison; the aox grid drops iteration for runtime reasons.)

Output: /Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100/
        strong_n10_phi19_N24_m1_v100.in

Run:
    python3 nitrogen/generate_nagao_ref.py
then on the Cloudy side:
    cd /Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100
    ../../source/cloudy.exe -r strong_n10_phi19_N24_m1_v100
"""

import os

OUTDIR = "/Users/jiamuh/c23.01/my_models/nagao_n10_phi19_N24_v100"
PREFIX = "strong_n10_phi19_N24_m1_v100"

CONTENT = f"""\
title BLR Nagao strong bump (matched to aox grid: N24, vturb 100)
set save prefix "{PREFIX}"
init "c84.ini"
interpolate (-8.0 -13.6) (-1.7 -1.0) (-1.3 1.48) (-1.0 2.0)
continue    (-0.19 1.65) (0.4 1.32) (1.87  -1.87)  (2.43 -2.37)
continue    (2.98 -2.79) (3.3 -2.96) (3.7 -3.38) (4.18 -4.05)
continue    (4.99 -5.59) (5.1 -6.67) (7.0 -12.37)
phi(H) 19
hden 10
metals 1
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
    in_path = os.path.join(OUTDIR, f"{PREFIX}.in")
    with open(in_path, "w") as f:
        f.write(CONTENT)
    print(f"Wrote {in_path}")
    print(f"Run with:")
    print(f"  cd {OUTDIR} && ../../source/cloudy.exe -r {PREFIX}")


if __name__ == "__main__":
    main()
