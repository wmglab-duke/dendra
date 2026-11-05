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
that require neither an accelerator nor the NEURON simulator, while `neuron and not
cuda` adds the required simulator-backed CPU tests without selecting GPU
comparisons.

Mark tests that inspect device metadata or mock compiler behavior with
`@pytest.mark.cpu` when they do not allocate or execute on an accelerator. This prevents a
parameter such as `device="cuda"` or `device="mps"` from assigning them to an accelerator lane.
Explicit accelerator markers and hardware guards still identify tests that
require an accelerator.

CUDA- and Triton-dependent cases use the `cuda` marker.
`neuron and cuda` narrows that lane to NEURON comparisons on CUDA. Longer
comparisons also carry `slow`, and randomized or property-generated cases
carry `stochastic`. Marker names are strict, so misspelled or undeclared
markers fail during test collection.

Apple MPS hardware tests use the separate `mps` marker. MPS guards and explicit
markers take precedence over inherited CPU metadata markers; an MPS guard that
mentions an Apple GPU is not classified as CUDA.

## Shorter development passes

Run the affected test module first. For a broader portable-CPU pass that omits
tests classified as slow, use:

```sh
python -m pytest tests -W error -m "cpu and not slow"
```

The `slow` compiler tests retain real Inductor/C++ execution, full recordings,
gradient checks, ownership checks, and repeated calls. They remain part of the
required CPU/NEURON lane above; the shorter pass is not a replacement for it.

The development dependencies include `pytest-xdist`. With enough memory for two
independent compiler processes, the complete lane can also use two workers:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 \
python -m pytest tests -n 2 --dist loadfile -W error -m "cpu or (neuron and not cuda)"
```

Keep worker counts conservative: compiler processes have independent memory
costs. Tests reset compiler caches between cases, without changing the cache
limits or reuse behavior being checked within a case.

For new functional tests, use Dynamo/eager capture for interface and scheduling
contracts, AOT-eager for reusable differentiable graph checks, and actual
Inductor for code-generation and native-kernel integration contracts. Raw eager
scan autograd still creates and partitions joint graphs, so it is not always
the cheaper choice for repeated checkpoint spans. Keep eager-specific coverage
where that execution path is the contract being tested.

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

## Run the MPS solver checks

On an Apple MPS host with an MPS-enabled `dendra-solvers` installation, verify
the backend and run the native solver integration cases with CPU fallback
disabled:

```sh
python -c "import dendra_solvers; assert dendra_solvers.is_mps_available()"
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest tests/test_mps_native_solvers.py -W error -m mps
```

These cases compare float32 native and functional trajectories and gradients
against CPU references, check owning topology-plan reuse, and exercise
full-graph atomic steps with the AOT-eager backend. `-m mps` selects all optional
MPS cases when running the complete test directory.

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

GitLab CI runs the same non-CUDA branch-coverage measurement, enforces a
ratcheted 84% global minimum, and retains JSON,
browsable HTML, and Cobertura reports. Critical modules also have individual
floors in `coverage-floors.toml`. After generating `coverage.json`, validate
those floors locally with:

```sh
python scripts/check_coverage_floors.py coverage.json
```

## Run the full lane as CI units

GitLab keeps ten physical jobs and installs dependencies once per job. The
required CPU/NEURON lane is divided into forty duration-balanced logical units,
with complete test modules kept together. Each job runs four units serially;
job 1 owns units 1, 11, 21 and 31. All forty units and the coverage aggregation
must qualify before merging. The functional compiler compatibility matrix is
an additional required check.

To run the same full lane locally with two concurrent unit processes:

```sh
python scripts/run_ci_shards.py --workers 2 --output-dir ci-results
```

`--workers` controls process concurrency independently of the forty logical
units. Keep it small enough for compiler memory use. Output and its sibling
runner directory must be Gitignored or outside the checkout, so generated
artifacts cannot change the source fingerprint. To run the four units
owned by GitLab job 1, add `--job-index=1`; its default worker count is one.

Every unit records its full selected inventory, assignments, outcomes, source
fingerprints and timings in `ci-results/<unit>/manifest.json`. Its command,
pytest log, JUnit report and raw branch coverage are retained alongside that
manifest. Runner and module-recovery evidence lives in `ci-results-runner/`.

The runner collects the current full lane before deciding whether previous
artifacts are reusable. Resume requires matching runtime, configuration,
environment and native-code fingerprints, unchanged assigned module hashes,
identical test inventory and assignments, successful execution of all assigned
IDs, and intact artifact hashes. Missing or invalid evidence reruns the unit.

With unchanged globals and test inventory, a recorded failure or an isolated
plain top-level test-function body edit reruns the complete affected module.
Checked recovery replaces its outcomes and coverage contexts; it does not rerun
only the failed ID or union stale coverage. Module-level, collection, helper and
class changes conservatively invalidate all units, as do runtime, shared test
support, conftest/configuration and other project changes. Added/removed tests
or changed assignments can require fresh units. Tests must restore shared state
and remain independent across modules; eligibility checks are not a Python
purity proof.

CI retains unit evidence in a branch/job cache, separately from the pip cache.
An eligible test-fix commit can reuse unchanged units even though its commit ID has
changed. Cache transport is best-effort: a missing cache causes normal execution,
and cache presence alone never proves correctness. Failed-attempt and recovery
provenance remain visible. The same restrictions apply to local resume.

Reuse is scoped to the same configured CPU/NEURON lane, image and device class.
Fingerprints cover Python/packages, native sources and binaries, and runner
options; they do not probe arbitrary environment variables or hardware.

After all forty units finish, validate their exact disjoint selected-ID
partition and combine raw statement/branch coverage with:

```sh
python scripts/combine_ci_coverage.py --shards 40 --input-dir ci-results --fail-under=84
```

The aggregator enforces the unchanged 84% global minimum and module floors in
`coverage-floors.toml`. Reports are written to `coverage.json`, `coverage.xml`
and `htmlcov/`. Per-unit coverage percentages are not qualification thresholds.

Merge-request pipelines and pushes to `develop` or `main` always run the full
checks. Feature-branch push pipelines select them when runtime, tests, build,
or CI files change. Documentation-only feature-branch pushes run the executable
documentation build; set `RUN_FULL_TESTS=1` to request full checks there too.
An open merge request replaces the duplicate feature push pipeline.

Unit estimates come from `scripts/ci_durations.json`; new modules use its mean
time per test. Manifests preserve profile hashes and actual duration records.
Forty units reduce retry scope, without pruning tests or changing the coverage
requirement. Module size, host speed, compilation caches, process contention,
collection and installation overhead still affect duration; this is not a
cold-CI timing prediction or a two-hour guarantee.
