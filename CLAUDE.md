# metallicity_gradient — project notes

## Working style — ASK WHEN NOT SURE

- **If you are not sure what the user wants, ASK — do not guess, do not start editing code or running things to "figure it out".** State briefly what you found / what's unclear, list the concrete options, and wait for the user to choose.
- This applies especially to: which folder to write to, whether to re-run something, what a prior decision/issue was, and any change to the analysis code.
- Don't go on long autonomous investigations. A short "here's what I see, here's the question" beats 5 minutes of silent digging.
- Report progress on anything that takes more than ~1 minute; never wait silently.
- This includes *formatting / style* choices, not just code. Before introducing a new structure (a list, a subsection, a new figure, a notation change), check whether it matches what the user wants — when in doubt, ask or just mirror the existing style.

## Python environment

- This project uses the conda env **`pypeit`**.
- When *Claude* runs anything itself (tests, debug scripts), use:
  `~/miniconda3/envs/pypeit/bin/python ...`
  (the bare `python3` in this shell is conda `base`, which is missing `emcee`, etc.)
- Commands handed to the *user* to run should still use plain `python3` — they have `pypeit` activated.
