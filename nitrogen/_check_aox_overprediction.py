"""Factor by which the solar-Z model over- or under-predicts the nitrogen/C IV
ratios inside the low-ion-consistent alpha_ox strip (Appendix A text check)."""
import numpy as np
from plot_aox_line_ratios import load_grid, low_ion_aox_range, observed_ratio, VDB, CSH

records = load_grid()
(lo, hi), log_z = low_ion_aox_range(records)
sel = sorted([r for r in records if np.isclose(r['logZ'], log_z)], key=lambda r: r['aox'])
aox = np.array([r['aox'] for r in sel])
strip = np.linspace(lo, hi, 50)
for key in ('NV', 'NIV', 'NIII'):
    model = np.interp(strip, aox, [r['lines'][key] / r['lines']['CIV'] for r in sel])
    for name, src in (('Vanden Berk', VDB), ('Constantin & Shields', CSH)):
        obs, _ = observed_ratio(key, 'CIV', source=src)
        if obs is not None:
            print(f'{key:5s} model {model.min():.4f}-{model.max():.4f}  {name:22s} obs {obs:.4f}'
                  f'  model/obs {model.min() / obs:.2f}-{model.max() / obs:.2f}')

# Hardest SED in the grid
for key in ('NV', 'NIII'):
    m = [r['lines'][key] / r['lines']['CIV'] for r in sel if np.isclose(r['aox'], -1.0)][0]
    print(f'alpha_ox = -1.0  {key}: model {m:.4f}  / VdB {m / observed_ratio(key, "CIV", VDB)[0]:.2f}'
          f'  / C&S {m / observed_ratio(key, "CIV", CSH)[0]:.2f}')
