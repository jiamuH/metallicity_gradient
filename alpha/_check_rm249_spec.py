import numpy as np
d = "/Users/jiamuh/sdssrm/rm249/"
for f in ("rm249_c4_w.dat", "rm249_si4_w.dat"):
    a = np.loadtxt(d + f)
    nz = a[:, 1] != 0
    w, fl = a[nz, 0], a[nz, 1]
    print(f, "nonzero window:", w.min(), "-", w.max(), "A, peak", fl.max(), "at", w[fl.argmax()])
a = np.loadtxt(d + "rm249_avg_w.dat")
w, blr = a[:, 0], a[:, 5]
for lo, hi, name in ((1350, 1450, "SiIV+OIV"), (1470, 1620, "CIV")):
    m = (w > lo) & (w < hi)
    print("avg BLR", name, ": peak", blr[m].max(), "at", w[m][blr[m].argmax()], "range", blr[m].min(), blr[m].max())
print("avg spectrum wavelength range:", w.min(), "-", w.max())
