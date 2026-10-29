"""Run the five fixed-nitrogen Cloudy grids for the SED / microturbulence
robustness test of the nitrogen gradient (referee comment 6), one after the
other, with a live progress bar in the terminal.

Progress is the number of finished models, counted from the per-model
grid*_<prefix>.out files that Cloudy writes while a grid runs (a model counts
as finished once its file ends with "Cloudy exited OK").  The time left is
estimated from the average time per finished model so far.

Run: python3 -m nitrogen.run_robustness_grids          (add --dry-run to only list the grids,
     or --grids PREFIX [PREFIX ...] to run other input files in the same folder)
"""
import argparse
import glob
import os
import subprocess
import sys
import time

CLOUDY = os.path.expanduser('~/c23.01/source/cloudy.exe')
MODEL_DIR = os.path.expanduser('~/c23.01/my_models/loc_metal_nitrogen')
PREFIXES = ['strong_LOC_m3_N23_turb0', 'strong_LOC_m3_N23_turb300',
            'AGN_aoxm1.40_LOC_m3_N23', 'AGN_aoxm1.65_LOC_m3_N23', 'AGN_aoxm1.90_LOC_m3_N23']
N_PER_GRID = 91          # 13 values of log phi x 7 values of log n_H
POLL_S = 5
BAR_W = 30


def n_finished(prefix, since, done):
    """Number of models of `prefix` finished since time `since`; `done` caches finished files."""
    for path in glob.glob(os.path.join(MODEL_DIR, f'grid*_{prefix}.out')):
        if path in done or os.path.getmtime(path) < since:
            continue
        with open(path, 'rb') as fh:
            fh.seek(max(os.path.getsize(path) - 4096, 0))
            if b'Cloudy exited OK' in fh.read():
                done.add(path)
    return len(done)


def fmt(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f'{h:d}:{m:02d}:{s:02d}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dry-run', action='store_true', help='list the grids and exit')
    ap.add_argument('--grids', nargs='+', default=PREFIXES, help='input-file prefixes to run')
    args = ap.parse_args()
    prefixes = args.grids
    missing = [p for p in prefixes if not os.path.exists(os.path.join(MODEL_DIR, p + '.in'))]
    if missing:
        sys.exit(f'missing input files in {MODEL_DIR}: {missing}')
    total = N_PER_GRID * len(prefixes)
    print(f'{len(prefixes)} grids x {N_PER_GRID} models = {total} models, in {MODEL_DIR}')
    if args.dry_run:
        for p in prefixes:
            print(f'  {p}.in')
        return

    t0, n_before = time.time(), 0
    for i, prefix in enumerate(prefixes, 1):
        start, done = time.time(), set()
        log = open(os.path.join(MODEL_DIR, f'{prefix}.runner.log'), 'w')
        proc = subprocess.Popen([CLOUDY, '-r', prefix], cwd=MODEL_DIR, stdout=log, stderr=subprocess.STDOUT)
        while True:
            running = proc.poll() is None
            n = N_PER_GRID if not running else min(n_finished(prefix, start, done), N_PER_GRID)
            n_all = n_before + n
            elapsed = time.time() - t0
            left = elapsed / n_all * (total - n_all) if n_all else float('nan')
            bar = '#' * int(BAR_W * n_all / total)
            sys.stdout.write(f'\r[{bar:<{BAR_W}s}] {n_all:3d}/{total} models | grid {i}/{len(prefixes)} '
                             f'{prefix} {n:2d}/{N_PER_GRID} | elapsed {fmt(elapsed)} | '
                             f'left ~{fmt(left) if n_all else "?:??:??"} ')
            sys.stdout.flush()
            if not running:
                break
            time.sleep(POLL_S)
        log.close()
        status = 'OK' if proc.returncode == 0 else f'exit code {proc.returncode}, see {prefix}.runner.log'
        sys.stdout.write(f'\n  finished {prefix} in {fmt(time.time() - start)} ({status})\n')
        n_before += N_PER_GRID
    print(f'All {len(prefixes)} grids done in {fmt(time.time() - t0)}.')


if __name__ == '__main__':
    main()
