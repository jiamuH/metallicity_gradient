# metallicity_gradient — project notes

Global rules (workflow, plotting, paper writing) live in ~/.claude/rules/ and apply here.

## Python environment
- conda env: `pypeit`.
- When Claude runs things itself: `~/miniconda3/envs/pypeit/bin/python ...`
  (bare `python3` in this shell is conda base, missing emcee etc.).
- Commands handed to the user: plain `python3` (pypeit already activated).
