"""Isolate: does the annotation bbox render? Scratch."""
import numpy as np
import alpha.loc_gradient_fit  # noqa: F401  (applies project rcParams)
import matplotlib.pyplot as plt

_bb = dict(boxstyle='round,pad=0.25', fc='black', ec='none', alpha=1.0)
fig, ax = plt.subplots(figsize=(7, 4))
ax.pcolormesh(np.ones((8, 8)), cmap='magma', vmin=0, vmax=1)   # bright bg
t = ax.text(0.03, 0.96, r'$p=0.35,\ \log r_{\rm ref}=17,\ Z_{\rm norm}=3\,Z_{\odot}$',
            transform=ax.transAxes, va='top', ha='left', color='w', bbox=_bb)
print('bbox patch:', t.get_bbox_patch())
fig.savefig('plots/alpha/loc_gradient_noA/_test_bbox.png', dpi=150, bbox_inches='tight')
print('saved _test_bbox.png')
