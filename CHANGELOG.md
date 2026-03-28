## v0.12.0 (2026-03-28)

### Feat

- add differentiable active and firing_rate analysis
- experimental: line sources (for extracellular stim)

### Fix

- correctly capture x,y,z segment coordinates for branched morphs

### Refactor

- refactor conduction_velocity and action_potential_width to use the same spike-time helper

## v0.11.1 (2026-03-06)

### Fix

- fix from_ascent classmethod for precomputed_interpolate_1d field
- fix rotate_azimuthal in Tree class

## v0.11.0 (2026-03-04)

### Feat

- differentiable chronaxie estimation

### Fix

- **stim/waveform.py**: fixes bi_rect_balanced

## v0.10.0 (2026-02-26)

### Feat

- **axonml/core.py**: str representation of Population and subclasses through extra_repr() (brief) and pretty() (verbose / complete)

## v0.9.0 (2026-02-24)

### Feat

- analysis tool to differentiably compute conduction velocity
