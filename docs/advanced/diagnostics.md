(diagnostics)=
# Diagnose model and performance problems

Use the default settings for ordinary simulations. Enable additional checks
when developing custom model code or investigating an incorrect result.
Checks diagnose specific contract violations; they do not establish that
arbitrary user code is correct for every input.

## Check the installation

`dn.doctor()` reports the installed runtime and solver capabilities. Include
that report, a minimal example, and the device/dtype when reporting a problem.
See {doc}`../installation` and {doc}`../gpu-deployment` for installation and
accelerator checks.

## Validate custom pure code

Construct a functional plan or runner inside `dn.ctx(CHECK_PURITY=True)` to
enable additional dependency, mutation, randomness and repeatability checks:

```python
with dn.ctx(CHECK_PURITY=True):
    program = dn.compile(model, dt=dt, steps=steps)
```

The plan retains this choice after the context exits. These diagnostics can
increase setup time. Ordinary lowering trusts the authored pure hooks, while
required shape, type, ownership and lifecycle checks remain active. Consult
{doc}`custom_models` when a hook needs evolving State or `CARRY` rather than
hidden mutable attributes.

For a current declared `AFFINE`, `MEMBRANE_ASSEMBLY="validate"` checks eligible
coefficient assembly. It can help diagnose an incorrect affine assertion or
coefficient method. The normal `"auto"` policy uses eligible affine assembly;
`"off"` selects ordinary current evaluation. See the
{doc}`mechanism reference <../api/dendra.models.mechanisms>` for current units
and the local-current contract.

## Respect initialized geometry and workspaces

Canonical Cable geometry is an initialized structural snapshot. Rebuild a
model from its Morphology when changing topology or frozen geometry. After
changing mutable solver-affecting parameters in native execution, reinitialize
the owning Population or Network when required. Explicit functional bindings
use fresh preparation after their source values change.

`RUNTIME_CONTRACT_VALIDATION="versioned"` is the default native policy. It
checks protected values at public execution boundaries and avoids repeatedly
reading unchanged tensor values. `"strict"` performs full frozen-value checks
and is useful when diagnosing unexpected mutation. `"initialize"` omits
repeated execution-boundary scans for trusted loops; use it only when your
application controls all writes and measurements justify it. It does not
disable construction, initialization or workspace-rebuild checks.

Avoid writes through `.data`, NumPy/DLPack aliases or raw storage. Those can
bypass PyTorch mutation tracking. Callbacks must not change protected geometry
or cached solver inputs during a run. Use documented input tapes for changing
stimulation, and explicit step boundaries for intentional application updates.

## Measure the execution you are using

Separate compilation/setup from steady execution when timing a program. Use
`program.execution_report()` after a run to inspect its selected dispatch.
Synchronize accelerators around timed regions. Compare identical dtype,
recording, inputs, training mode and model sizes.

Use scan for gradients and while for inference. Smaller chunks can bound
temporary stimulus and callback segments, but requested in-memory recordings
still need storage. Activation checkpointing trades backward recomputation for
saved activations; it does not truncate the gradient horizon. See
{doc}`functional_execution`, {doc}`training_and_parameters` and
{doc}`recording_and_callbacks` before changing execution settings.

Inference-created tensors lack ordinary mutation counters. Prefer constructing
models outside `torch.inference_mode()` and choose inference execution on the
runner. See {doc}`../misc/M1_network_performance_patterns` for model-specific
performance examples.
