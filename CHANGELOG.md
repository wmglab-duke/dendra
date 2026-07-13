## v0.17.0 (2026-07-10)

### Feat

- add dendra.step to advance state by one timestep
- euler-maruyama SDE for States
- register new random distributions for use with RAND declarations in Parameterized
- add RAND variables to Parameterized
- expose torch.compile options through dendra
- PointProcess slots

### Fix

- make duration-based Population and Network runs composable by retaining incomplete fixed timesteps across calls

## v0.16.0 (2026-06-24)

### Feat

- differentiable NetStim event scheduler
- models now pickleable
- new differentiable Network backends for better memory efficiency
- universal cache & load state in Network
- add DTYPE and DEVICE to ctx
- bitpacked history netcon backend
- implement ClearanceProcess, ClampProcess, ExchangeProcess
- add DiffusionProcess for materials
- add bufferimplicit, linearimplicit, rosenbrock ODE solvers
- implemented differentiable fire and apcount mechanisms
- add paired-pulse and sds metric to analysis and non-differentiable versions
- add ability for mech delay buffers to be time-major
- batched delays in Mechanism
- continuous / analog synapses added
- add ability to stimulate mechanisms directly using Waveforms

### Fix

- fixes interaction between state-cache restore and the new source-history training backend; fixes check_weight_shape
- fixes steady_state() for models
- robust dynamic code compiler (fix for py3.13)
- more robust Slice __setattr__
- fix network and model batching

### Refactor

- cpp option for bitpacked netocon backend
- replace cuda bitpack netcon kernels with triton
- rename mod files
- NetCon storing event queues now optional
- remove redundant ion_style params
- AxModule -> DNModule

### Perf

- reduce peak-memory thrashing for network re-initialization
- netcon dense backend optimizations
- v1 sparse netcon backend to improve memory efficiency of Networks

## v0.15.0 (2026-05-20)

### Feat

- **models.fields.precomputed**: added mesh-aware field interpolators to interface with SimNIBS

### Fix

- **instrument.thresholder**: fix to handle nans in field for Thresholder
- **models.tree**: fixes rotate_into_direction

### Refactor

- rename package dendra
- rename package from axonml to dendra

## v0.14.0 (2026-04-20)

### Feat

- align network connection semantics with NEST

## v0.13.0 (2026-04-08)

### Feat

- **dendra.models.callbacks**: adding t_end_check to ThresholdCallbacks to limit threshold detection to specified time window
- **dendra.models.callbacks**: added ActiveALCount to determine activity based on total # registered APs

### Fix

- **dendra.models.callbacks**: add t_end_check as param to Active* threshold callback family
- resolve positive and negative parameter overrides in mro for Parameterized

### Refactor

- **models.parametric**: refactor resolve method to call general resolve function

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

- **dendra/core.py**: str representation of Population and subclasses through extra_repr() (brief) and pretty() (verbose / complete)

## v0.9.0 (2026-02-24)

### Feat

- analysis tool to differentiably compute conduction velocity
