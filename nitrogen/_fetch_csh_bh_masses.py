"""Look up black-hole masses and Eddington ratios of the 22 Constantin & Shields
(2003) NLS1s in VizieR catalogs. Prints the matching rows of every table of
each catalog, so the columns can be inspected before anything is adopted.

Run: ~/miniconda3/envs/pypeit/bin/python nitrogen/_fetch_csh_bh_masses.py J/ApJS/187/64
"""
import io
import re
import sys
import urllib.parse
import urllib.request

# Constantin & Shields (2003) Table 2: name, z, log L_nu(1450) [H0=50, q0=0.5]
CSH_SAMPLE = [
    ('Ark 564', 0.0239, 28.25), ('1H 0707-495', 0.0355, 29.01),
    ('IRAS 13224-3809', 0.0580, 29.03), ('IRAS 13349+2438', 0.1038, 28.55),
    ('KUG 1031+398', 0.0432, 28.10), ('Mrk 110', 0.0340, None),
    ('Mrk 335', 0.0260, 29.23), ('Mrk 478', 0.0755, 29.86),
    ('Mrk 486', 0.0387, 27.84), ('Mrk 493', 0.0311, 28.35),
    ('Mrk 766', 0.0127, 26.13), ('NGC 4051', 0.00245, 26.38),
    ('PG 1211+143', 0.0804, 29.75), ('PG 1351+640', 0.0875, 29.65),
    ('PG 1404+226', 0.0935, 29.22), ('PG 1411+442', 0.0875, 29.68),
    ('PG 1444+407', 0.2640, 30.55), ('RX J0134-42', 0.2360, 30.31),
    ('Ton S 180', 0.0613, 29.63), ('WPVS 007', 0.0280, 27.78),
    ('I Zw 1', 0.0585, 29.40), ('II Zw 136', 0.0619, 29.59),
]


def key(name):
    """Name reduced to lower-case letters and digits, e.g. 'tons180'."""
    return re.sub(r'[^a-z0-9]', '', name.lower())


def match(row_name, keys):
    k = key(row_name)
    for name, kk in keys.items():
        # catalogs abbreviate declinations (RX J0134-42 vs RX J0134.2-4258)
        if k == kk or (len(kk) > 6 and (k.startswith(kk) or kk.startswith(k))):
            return name
    return None


def fetch(source):
    url = ('https://vizier.cds.unistra.fr/viz-bin/asu-tsv?'
           + urllib.parse.urlencode({'-source': source, '-out.max': 'unlimited',
                                     '-out.all': '', '-oc.form': 'dec'}))
    with urllib.request.urlopen(url, timeout=60) as f:
        return f.read().decode('latin-1')


def main():
    keys = {n: key(n) for n, _, _ in CSH_SAMPLE}
    for source in sys.argv[1:]:
        text = fetch(source)
        for block in text.split('#Table')[1:]:
            lines = block.splitlines()
            table = lines[0].strip()
            data = [l for l in lines[1:] if l and not l.startswith('#')]
            if len(data) < 3:
                continue
            header, rows = data[0].split('\t'), data[3:]   # skip units and dashes
            name_cols = [i for i, h in enumerate(header) if h.strip() in ('Name', 'Object', 'ID')]
            if not name_cols:
                print(f'{table}: no name column, columns {header}')
                continue
            hits = [(match(r.split('\t')[name_cols[0]], keys), r) for r in rows]
            hits = [(n, r) for n, r in hits if n]
            print(f'\n=== {table}  ({len(hits)} matches)')
            print('\t'.join(header))
            for n, r in hits:
                print(r)


if __name__ == '__main__':
    main()
