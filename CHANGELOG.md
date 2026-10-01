## v0.27.1 (2026-10-01)

### Fix

- **docs**: execute notebooks in documentation CI

## v0.27.0 (2026-10-01)

### Feat

- **packaging**: add GitHub trusted publishing
- **analysis**: add hard-forward, branch-conditioned trace descriptors and validated threshold-derived objectives for model training. package the analysis API into focused modules, document complete training workflows, and add end-to-end examples and gradient checks

### Fix

- **initialization**: rematerialize effective parameter buffers after parameter-writing initialization transforms for repeated initialized simulations remain connected to trainable raw parameters

## v0.26.1 (2026-09-24)

### Fix

- **passive_end_nodes**: optionally adjust diam in passive_end_nodes_

## v0.26.0 (2026-09-10)

### Feat

- **models**: passive_end_nodes_
- **func**: chunked compilation

## v0.25.0 (2026-09-04)

### BREAKING CHANGE

- new canonical Mechanism and State declaration protocol

### Feat

- **func**: add composable functional population execution
- **functional**: implement DERIVED_BUFFER
- **solvers**: add vmap functionality to custom solvers
- **runtime**: add MPS support and canonical support registry

### Fix

- **networks**: network conat functional across scalar neuron types

### Refactor

- **torchinductor**: make dendra torchinductor patches opt-in
- **mechanisms**: preserve population axis

## v0.24.0 (2026-08-16)

### Feat

- **stimulation**: add functionality to insert / remove intrinsic activity
- **mechanisms**: add opt-in population-shaped storage for distributed mechanisms on shared compartment columns
- **mechanisms**: expose the population-axis layout as a construction-scoped `dendra.ctx` / environment policy with explicit constructor overrides and audited model-family defaults

### Fix

- **mechanisms**: make support-aware copying, batching, device moves, deletion, and checkpoint validation fail closed
- **initialization**: replace the hidden double mechanism initialization with one ordered transaction; run `INITIAL` once, keep per-step material sources and transport out of `t=0`, retain post-guard values, publish final ionic currents after concentration and Nernst synchronization, invalidate steady snapshots when `v_init` changes, and consume post-restore solver rebuilds only once

### Perf

- **mechanisms**: reuse exact-support gathers across initialization, state advancement, geometry and field synchronization, and reduce compatible current contributions before scattering

## v0.23.1 (2026-08-15)

### Fix

- **copies**: parameters in mechanism copies now broadcast correctly

## v0.23.0 (2026-08-14)

### Feat

- **mechanisms**: handle voltage-independent currents returning 0 scalar conductance
- **materials**: support regional intracellular/extracellular diffusion with named finite-volume geometry on unbranched and Tree populations, edge-local diffusivity, and implicit spatially varying reservoir coupling
- **materials**: allow `DiffusionProcess` insertion on compartment subsets; selected compartments form an induced diffusion graph with sealed crossing edges and independent disconnected components
- **morphology**: connect two morphologies

### Refactor

- **mechanisms**: detect mechanisms with identical ordered compartment support and gather voltage once per support

## v0.22.0 (2026-08-07)

### Feat

- **core**: delete mechanisms

### Fix

- **slice**: expose flat_index for slice
- **fields**: field and thresholder updated to work with batched models
- **ions**: fix stale ion currents in mechanisms that read ionic currents

### Refactor

- specialized step_population and step_network

## v0.21.1 (2026-07-17)

### Fix

- **visualization**: interactive 3d morphology plot

## v0.21.0 (2026-07-17)

### Feat

- **morphologies**: delete sections; refine discretization using d_lambda rule
- **morphologies**: edit & visualize morphologies

## v0.20.0 (2026-07-15)

### Feat

- **models**: implemented Cable class to consume unbranched native morphology
- **morpholgy**: v1 native morphology spec

## v0.19.0 (2026-07-13)

### Feat

- **slice**: commuting Slices & explicit intersection

## v0.18.1 (2026-07-13)

### Fix

- **slice**: preserve mechanism parameterizations across rebuilds
- **slice**: define safe indexing mutation and lifecycle semantics
- **batching**: standardize and document dendra stim batching semantics
- **solvers**: harden implicit and symbolic solver correctness
- **networks**: make state cache translation exact across runtimes

## v0.18.0 (2026-07-13)

### Feat

- fractional timsteps carry over across run boundaries

### Fix

- **networks**: make mode transitions and checkpoints exact
- **mechanisms**: rebind temperature state on initialization
- **models**: preserve float64 precision during model construction
- improve symbolic diff robustness
- fix Exp2Syn, Mechanism.rename, and network-clock drift
- fix extracellular stim for Tree
- make sliding window averaging suitable for older CPU hardware

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
