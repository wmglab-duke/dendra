(gpu-deployment)=
# GPU deployment

Dendra's GPU solvers are included in the main package. Their availability
depends on the PyTorch build and NVIDIA driver installed on the host. If you
need a particular CUDA build of PyTorch, install it before Dendra as described
in {ref}`choose-pytorch-build`.

## Inspect the deployment

After installing Dendra, run the deployment doctor and require a working CUDA
runtime:

```sh
dendra doctor --require-cuda
# Equivalent when the console script is not on PATH:
python -m dendra doctor --require-cuda
```

The default doctor is inspection-only. It reports the PyTorch and CUDA
runtimes, visible GPUs, NVCC and C++ compiler discovery, CUDA-version
alignment, packaged native sources, selected GPU architectures, cache
writability, and matching cached artifacts. It does not compile, load, or
launch a native extension. With `--require-cuda`, the command exits with an
error when CUDA or PyTorch CUDA support is unavailable.

## Probe the optional native NetCon extension

Dendra can use an optional native CUDA extension for packed NetCon event
delivery. To test the complete native path, opt into a JIT build followed by a
small end-to-end kernel smoke test:

```sh
dendra doctor --require-cuda --probe-native-bitpack
```

The probe can invoke the local CUDA and C++ toolchains and populate PyTorch's
extension cache. Run it when validating a deployment that is expected to use
the native event-delivery path.

## Choose the fallback policy

If the optional NetCon CUDA extension cannot load or launch, Dendra uses its
correct pure-PyTorch implementation by default. A deployment that must report
or reject this performance-path change can select a stricter policy around the
relevant operation:

```python
import dendra as dn

with dn.ctx(NATIVE_EXTENSION_POLICY="require"):
    network.run(...)
```

The supported policies are:

- `"fallback"` (default): use the portable implementation silently;
- `"warn"`: emit a warning before using the portable implementation; and
- `"require"`: raise an error instead of falling back.

Set the `NATIVE_EXTENSION_POLICY` environment variable before importing
Dendra to apply a policy process-wide, for example:

```sh
NATIVE_EXTENSION_POLICY=require python your_simulation.py
```

The policy is enforced only when an eligible native CUDA path is attempted.
CPU execution and dense-backend execution are unaffected.
