"""Recover the Constantin & Shields (2003) NLS1 average UV composite from the
vector IDL EPS of their Figure 4 (arXiv astro-ph/0302369, f4.eps).

IDL operators: M = moveto, R = rlineto, P = lineto, N = rmoveto, D = stroke.
Data curves are drawn at linewidth 15; axes at 37.5.

Step 1 (this script, --inspect): list the panel frames, the y-axis major
ticks and the long polylines, so the calibration can be checked by eye.
Step 2: write the calibrated top-panel spectrum to data/nitrogen/.

Run (from the project root): python3 nitrogen/_extract_cs03_composite.py [--inspect]
"""
import sys

import numpy as np

EPS = ('/private/tmp/claude-501/-Users-jiamuh-python-metallicity-gradient/'
       '1bb174d7-7361-425a-9ae9-bacf96eb632a/scratchpad/arx_0302369/f4.eps')
OUT = 'data/nitrogen/cs03_nls1_average_composite.txt'

# x calibration from the labelled major ticks (1500, 2000, 2500 A)
X_TICKS_PS = (6606, 10948, 15289)
X_TICKS_A = (1500.0, 2000.0, 2500.0)


def parse(path=EPS):
    """Return a list of (linewidth, polyline Nx2 array) in PostScript units."""
    toks = []
    for line in open(path, encoding='latin-1'):
        if line.startswith('%'):
            continue
        toks.extend(line.split())
    stack, lines = [], []
    lw, cur, poly = None, None, []

    def flush():
        if len(poly) > 1:
            lines.append((lw, np.array(poly, float)))

    for t in toks:
        if t == 'setlinewidth':
            lw = float(stack.pop()); stack.clear()
        elif t == 'M':
            flush(); y, x = stack.pop(), stack.pop(); stack.clear()
            cur = (float(x), float(y)); poly = [cur]
        elif t == 'R':
            dy, dx = stack.pop(), stack.pop(); stack.clear()
            cur = (cur[0] + float(dx), cur[1] + float(dy)); poly.append(cur)
        elif t == 'P':
            y, x = stack.pop(), stack.pop(); stack.clear()
            cur = (float(x), float(y)); poly.append(cur)
        elif t == 'N':
            flush(); dy, dx = stack.pop(), stack.pop(); stack.clear()
            cur = (cur[0] + float(dx), cur[1] + float(dy)); poly = [cur]
        elif t == 'D':
            flush(); poly = [cur] if cur else []
        else:
            try:
                float(t); stack.append(t)
            except ValueError:
                stack.clear()
    flush()
    return lines


def x_to_wave(x):
    a, b = np.polyfit(X_TICKS_PS, X_TICKS_A, 1)
    return a * np.asarray(x) + b


def inspect():
    lines = parse()
    frames = sorted({round(p[0, 1]) for lw, p in lines
                     if lw == 37.5 and len(p) == 2 and p[0, 0] == 3429 and p[1, 0] == 19431})
    print('horizontal frame lines at y =', frames)
    # left-axis major ticks (length 320) give the y-tick positions per panel
    ticks = sorted({round(p[0, 1]) for lw, p in lines
                    if lw == 37.5 and len(p) == 2 and p[0, 0] == 3429 and p[1, 0] - p[0, 0] == 320})
    print('left major ticks at y =', ticks)
    data = [(lw, p) for lw, p in lines if lw == 15 and len(p) > 50]
    for lw, p in data:
        print(f'  data polyline: {len(p):5d} points, x {p[:, 0].min():.0f}-{p[:, 0].max():.0f}'
              f' (wave {x_to_wave(p[:, 0].min()):.0f}-{x_to_wave(p[:, 0].max()):.0f} A),'
              f' y {p[:, 1].min():.0f}-{p[:, 1].max():.0f}')


# top panel: frame y 10160 (flux 0) to 18288 (flux 8)
Y0_PS, Y_PER_UNIT = 10160.0, (18288.0 - 10160.0) / 8.0


def extract():
    """Top-panel (average composite) spectrum: wavelength [A], F_lambda
    (normalised to unit mean flux over 1430-1470 A).  Points clipped at the
    top of the panel (Ly-alpha peak) are dropped."""
    import os
    pts = np.vstack([p for lw, p in parse() if lw == 15 and len(p) > 50
                     and p[:, 1].min() >= Y0_PS])
    pts = pts[np.argsort(pts[:, 0])]
    _, keep = np.unique(pts[:, 0], return_index=True)      # drop repeated segment joints
    pts = pts[keep]
    pts = pts[pts[:, 1] < 18287]                           # clipped at the panel top
    wave, flux = x_to_wave(pts[:, 0]), (pts[:, 1] - Y0_PS) / Y_PER_UNIT
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savetxt(OUT, np.column_stack([wave, flux]), fmt='%.2f %.4f',
               header='Constantin & Shields (2003) NLS1 average composite, recovered from '
                      'the vector EPS of their Figure 4 (top panel).\n'
                      'rest wavelength [A]   F_lambda (unit mean over 1430-1470 A)')
    sel = (wave > 1430) & (wave < 1470)
    print(f'{len(wave)} points, {wave.min():.1f}-{wave.max():.1f} A, median step '
          f'{np.median(np.diff(wave)):.3f} A; mean flux 1430-1470 A = {flux[sel].mean():.3f}')
    print(f'Saved {OUT}')


if __name__ == '__main__':
    if '--inspect' in sys.argv:
        inspect()
    else:
        extract()
