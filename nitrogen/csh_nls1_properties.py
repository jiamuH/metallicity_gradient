"""Black-hole masses and Eddington ratios of the 21 Constantin & Shields (2003)
NLS1s that contribute to the nitrogen-line composite (Mrk 110 is left out: its
only HST spectrum covers Ly-alpha and none of the lines used).

Inputs
  z, log L_nu(1450)  Constantin & Shields (2003) Table 2, computed by them for
                     H0 = 50, q0 = 0.5, Lambda = 0; converted here to flat
                     LambdaCDM (H0 = 70, Omega_m = 0.3).
  FWHM(H-beta)       Leighly (1999b) Table 1 [L99]; Boroson & Green (1992)
                     Table 2 [BG92]; Grupe et al. (1999) Table 1 [G99];
                     Veron-Cetty et al. (2001) as listed by Wang & Lu (2001) [VVG01].
  RM masses          AGN Black Hole Mass Database (Bentz & Katz 2015).

Method
  lambda L_lambda(5100) is extrapolated from L_nu(1450) with one power law
  F_nu ~ nu^alpha, alpha = -0.79 (the Constantin & Shields median composite).
  For the ten objects Constantin & Shields flag as UV-absorbed, published
  lambda L_lambda(5100) values replace the extrapolation (PUBLISHED_L5100).
  Single-epoch masses use the Vestergaard & Peterson (2006) H-beta relation.
  L_bol = 9.26 lambda L_lambda(5100) for all objects (Richards et al. 2006;
  Shen et al. 2008); L_Edd = 1.26e38 (M / M_sun) erg/s.

Run: python3 -m nitrogen.csh_nls1_properties   (from the project root)
"""
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

ALPHA_NU = -0.79
C_KMS = 2.998e5

# name, z, log L_nu(1450) [CS03 cosmology], FWHM(Hb) km/s, FWHM source, RM log M (or None)
SAMPLE = [
    ('Ark 564',          0.0239,  28.25,  950, 'L99',   None),
    ('1H 0707-495',      0.0355,  29.01, 1050, 'L99',   None),
    ('IRAS 13224-3809',  0.0580,  29.03,  650, 'L99',   None),
    ('IRAS 13349+2438',  0.1038,  28.55, 2200, 'L99',   None),
    ('KUG 1031+398',     0.0432,  28.10,  840, 'L99',   None),
    ('Mrk 335',          0.0260,  29.23, 1640, 'L99',   7.230),
    ('Mrk 478',          0.0755,  29.86, 1450, 'L99',   None),
    ('Mrk 486',          0.0387,  27.84, 1480, 'BG92',  None),
    ('Mrk 493',          0.0311,  28.35,  740, 'VVG01', None),
    ('Mrk 766',          0.0127,  26.13,  850, 'L99',   6.822),
    ('NGC 4051',         0.00245, 26.38, 1150, 'L99',   5.891),
    ('PG 1211+143',      0.0804,  29.75, 1860, 'L99',   None),
    ('PG 1351+640',      0.0875,  29.65, 5660, 'BG92',  None),
    ('PG 1404+226',      0.0935,  29.22,  880, 'L99',   None),
    ('PG 1411+442',      0.0875,  29.68, 2670, 'BG92',  8.539),
    ('PG 1444+407',      0.2640,  30.55, 2480, 'BG92',  None),
    ('RX J0134-42',      0.2360,  30.31, 1160, 'G99',   None),
    ('Ton S 180',        0.0613,  29.63,  980, 'L99',   None),
    ('WPVS 007',         0.0280,  27.78, 1620, 'G99',   None),
    ('I Zw 1',           0.0585,  29.40, 1240, 'L99',   None),
    ('II Zw 136',        0.0619,  29.59, 2330, 'BG92',  7.433),
]


# Published lambda L_lambda(5100) for the objects Constantin & Shields flag as
# UV-absorbed (their 1450 A fluxes are lower limits). Values are
# log lambda L_lambda [erg/s] for H0 = 70, or ('flux', F_lambda(5100) [erg/s/cm^2/A])
# to be converted here. 1H 0707-495 (Done & Jin 2016) and RX J0134-42 (Jin et
# al. 2023) are quoted for H0 = 70 cosmologies; NGC 4051 and Mrk 766 are
# host-corrected (Bentz et al. 2013); Ark 564 is a mean over 91 epochs without
# host correction (Shapovalova et al. 2012).
PUBLISHED_L5100 = {
    'NGC 4051':     (41.96,             'Bentz et al. 2013'),
    'Mrk 766':      (42.51,             'Bentz et al. 2013'),
    'PG 1351+640':  (44.835,            'Vestergaard & Peterson 2006'),
    'PG 1404+226':  (44.379,            'Vestergaard & Peterson 2006'),
    'PG 1411+442':  (44.621,            'Vestergaard & Peterson 2006'),
    'Mrk 486':      (43.961,            'Vestergaard & Peterson 2006'),
    '1H 0707-495':  (np.log10(3.3e43),  'Done & Jin 2016'),
    'RX J0134-42':  (np.log10(8.86e44), 'Jin et al. 2023'),
    'Ark 564':      (('flux', 5.07e-15), 'Shapovalova et al. 2012'),
    'WPVS 007':     (('flux', 1.1e-14),  'Leighly et al. 2009'),
}
BC_5100 = 9.26   # Richards et al. (2006) / Shen et al. (2008) bolometric correction


def dl_old_cm(z, h0=50.0):
    """Luminosity distance for q0 = 0.5, Lambda = 0 (Mattig relation)."""
    return 2 * C_KMS / h0 * ((1 + z) - np.sqrt(1 + z)) * u.Mpc.to(u.cm)


def compute(alpha_nu=ALPHA_NU, use_published=True):
    """With use_published, the UV-absorbed objects take their published
    lambda L(5100); all others extrapolate it from L_nu(1450). Every object
    uses L_bol = BC_5100 lambda L(5100)."""
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    rows = []
    for name, z, lnu_old, fwhm, src, log_m_rm in SAMPLE:
        d_new = cosmo.luminosity_distance(z).to(u.cm).value
        log_lnu = lnu_old + 2 * np.log10(d_new / dl_old_cm(z))
        nu1450, nu5100 = C_KMS * 1e13 / 1450, C_KMS * 1e13 / 5100   # Hz
        log_l1450 = log_lnu + np.log10(nu1450)                         # lambda L_lambda
        log_l5100_uv = log_lnu + alpha_nu * np.log10(nu5100 / nu1450) + np.log10(nu5100)
        log_l5100, l_src = log_l5100_uv, 'UV'
        if use_published and name in PUBLISHED_L5100:
            val, l_src = PUBLISHED_L5100[name]
            if isinstance(val, tuple):       # rest-frame F_lambda(5100) -> lambda L_lambda
                val = np.log10(4 * np.pi * d_new**2 * val[1] * 5100 * (1 + z))
            log_l5100 = val
        log_lbol = np.log10(BC_5100) + log_l5100
        log_m_se = 6.91 + 2 * np.log10(fwhm / 1e3) + 0.5 * (log_l5100 - 44)
        log_m = log_m_rm if log_m_rm is not None else log_m_se
        log_edd = log_lbol - np.log10(1.26e38) - log_m
        rows.append(dict(name=name, z=z, log_l1450=log_l1450, log_l5100_uv=log_l5100_uv,
                         log_l5100=log_l5100, l_src=l_src, fwhm=fwhm, src=src,
                         log_m_se=log_m_se, log_m_rm=log_m_rm, log_m=log_m, log_edd=log_edd))
    return rows


FWHM_REF = {'L99': 'Leighly 1999', 'BG92': 'Boroson \\& Green 1992',
            'G99': 'Grupe et al.\\ 1999', 'VVG01': 'V\\\'eron-Cetty et al.\\ 2001'}
RM_REF = {
    'Mrk 335': 'Peterson et al.\\ 1998, 2004; Grier et al.\\ 2012; Du et al.\\ 2014',
    'NGC 4051': 'Denney et al.\\ 2009; Fausnaugh et al.\\ 2017',
    'Mrk 766': 'Bentz et al.\\ 2009',
    'PG 1411+442': 'Kaspi et al.\\ 2000; Peterson et al.\\ 2004',
    'II Zw 136': 'Grier et al.\\ 2008, 2012',
}


def latex_rows():
    """Rows for Table R1 of the reply letter."""
    for r in compute():
        name = r['name'].replace('-', '$-$')
        l_src = 'UV' if r['l_src'] == 'UV' else r['l_src'].replace('et al.', 'et al.\\').replace('&', '\\&')
        if r['log_m_rm'] is not None:
            m = f"{r['log_m']:.2f}$^{{\\rm RM}}$"
            m_src = RM_REF[r['name']]
        else:
            m, m_src = f"{r['log_m']:.2f}", 'single-epoch'
        print(f"{name} & {r['z']:.3f} & {r['fwhm']} & {FWHM_REF[r['src']]} & {r['log_l5100']:.2f} & {l_src}"
              f" & {m} & {m_src} & ${r['log_edd']:.2f}$ \\\\")


def main():
    import sys
    if '--latex' in sys.argv:
        latex_rows()
        return
    rows = compute()
    print(f"{'object':17s} {'z':>7s} {'logL1450':>8s} {'L5100uv':>7s} {'L5100':>6s} {'FWHM':>5s} {'src':>5s}"
          f" {'logM_SE':>7s} {'logM_RM':>7s} {'logM':>6s} {'logEdd':>6s}  L5100 source")
    for r in rows:
        rm = f"{r['log_m_rm']:7.2f}" if r['log_m_rm'] is not None else '      -'
        print(f"{r['name']:17s} {r['z']:7.4f} {r['log_l1450']:8.2f} {r['log_l5100_uv']:7.2f} {r['log_l5100']:6.2f}"
              f" {r['fwhm']:5d} {r['src']:>5s} {r['log_m_se']:7.2f} {rm} {r['log_m']:6.2f} {r['log_edd']:6.2f}  {r['l_src']}")
    for key in ('z', 'log_l5100', 'log_m', 'log_edd'):
        v = np.array([r[key] for r in rows])
        print(f'{key:10s} range {v.min():6.2f} to {v.max():6.2f}, median {np.median(v):6.2f}')
    se_rm = [(r['log_m_se'] - r['log_m_rm']) for r in rows if r['log_m_rm'] is not None]
    print('single-epoch minus RM mass for the RM objects:', np.round(se_rm, 2))


if __name__ == '__main__':
    main()
