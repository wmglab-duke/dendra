(testing-and-coverage)=
# Testing and code coverage

This guide describes Dendra's test lanes, accelerator checks, and coverage
policy. Run these commands from the root of a source checkout.

## Set up the test environment

Install Dendra in editable mode with the development dependencies and the
recommended native CPU solvers:

```sh
python -m pip install --editable ".[dev,solvers]"
```

The development dependencies include `pytest` and `pytest-cov`. If no
compatible `dendra-solvers` distribution is available for your platform, use
`python -m pip install --editable ".[dev]"`; unbranched CPU cables can use
Dendra's PyTorch fallback.

To run the complete test suite for the capabilities available on the current
host, use:

```sh
python -m pytest tests
```

## Run the required CPU and NEURON lane

Tests are classified into execution lanes. The required CI lane combines the
portable CPU suite with every non-CUDA NEURON integration and reference test:

```sh
python -m pytest tests -W error -m "cpu or (neuron and not cuda)"
```

This expression runs each selected test once. The `cpu` marker covers tests
that require neither CUDA nor the NEURON simulator, while `neuron and not
cuda` adds the required simulator-backed CPU tests without selecting GPU
comparisons.

CUDA- and Triton-dependent cases use the `cuda` marker.
`neuron and cuda` narrows that lane to NEURON comparisons on CUDA. Longer
comparisons also carry `slow`, and randomized or property-generated cases
carry `stochastic`. Marker names are strict, so misspelled or undeclared
markers fail during test collection.

## Run the CUDA lane

On a CUDA and Triton worker, verify the runtime and run the accelerator lane
independently:

```sh
python -c "import torch, triton; assert torch.cuda.is_available()"
python -m pytest tests -W error -m cuda
```

See the [GPU deployment guide](gpu-deployment.md) for environment diagnostics
and the optional native-extension probe.

### Check native CUDA kernels with Compute Sanitizer

The native NetCon CUDA kernels have a dedicated NVIDIA Compute Sanitizer lane:

```sh
python scripts/run_cuda_sanitizers.py
```

This command runs memcheck, racecheck, initcheck, and synccheck with source
line information. For reliable initcheck results, the CUDA toolkit used to
compile the native extension must have the same major version as PyTorch's
CUDA runtime. The runner detects mismatches, skips initcheck in its all-tools
lane, and explains how to restore that coverage.

On WSL with a WDDM GPU, sanitizer availability can depend on the Windows
driver and debugging-interface support. See NVIDIA's
[operating-system-specific Compute Sanitizer documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#operating-system-specific-behavior).

## Measure branch and statement coverage

Accelerator coverage artifacts are kept separate from the required CPU and
NEURON percentage. Kernel correctness is enforced primarily through dense
numerical oracles, gradient checks, boundary-shape contracts, and backend
equivalence tests. The required CPU report uses `.coveragerc.cpu` to omit only
the seven accelerator-only Triton kernel bodies. Their contracts, dispatch and
fallback paths, network Triton operations, and GPU diagnostics remain in its
coverage denominator.

Run the required lane with statement and branch coverage, print uncovered line
numbers, and generate JSON and browsable HTML reports with:

```sh
python -m pytest tests -W error -m "cpu or (neuron and not cuda)" --cov=dendra --cov-branch --cov-config=.coveragerc.cpu --cov-report=term-missing:skip-covered --cov-report=json:coverage.json --cov-report=html
```

The `TOTAL` row is the overall coverage result. Open `htmlcov/index.html` to
inspect coverage by module and identify untested lines and branches. The
required lane installs and exercises NEURON and the CPU solvers. Accelerator
execution remains outside this measurement, while CPU-testable CUDA and Triton
interfaces remain covered.

For test-impacting changes, GitLab CI runs the same non-CUDA branch-coverage
measurement, enforces a ratcheted 84% global minimum, and retains JSON,
browsable HTML, and Cobertura reports. Critical modules also have individual
floors in `coverage-floors.toml`. After generating `coverage.json`, validate
those floors locally with:

```sh
python scripts/check_coverage_floors.py coverage.json
```
