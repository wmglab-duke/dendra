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

Mark tests that inspect device metadata or mock compiler behavior with
`@pytest.mark.cpu` when they do not allocate or execute on CUDA. This prevents a
parameter such as `device="cuda"` from assigning them to the CUDA lane.
Explicit `cuda` markers and CUDA/GPU hardware guards still identify tests that
require an accelerator.

CUDA- and Triton-dependent cases use the `cuda` marker.
`neuron and cuda` narrows that lane to NEURON comparisons on CUDA. Longer
comparisons also carry `slow`, and randomized or property-generated cases
carry `stochastic`. Marker names are strict, so misspelled or undeclared
markers fail during test collection.

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

## Run the full lane as CI shards

GitLab CI splits the required CPU/NEURON lane across ten jobs so that each job
has its own timeout. Together they execute the same selected test set as the
single-process command above. All ten jobs and the coverage aggregation job
must pass before merging. The functional compiler compatibility matrix remains
an additional required check.

Merge-request pipelines and pushes to `develop` or `main` always run the full
checks. Feature-branch push pipelines select them when runtime, tests, build,
or CI files change. Documentation-only feature-branch pushes run the executable
documentation build; set the CI/CD variable `RUN_FULL_TESTS=1` to request the
full checks on those pushes too.
When a feature branch has an open merge request, its merge-request pipeline
replaces the duplicate push pipeline.

Each `test_shards` job records its test inventory and durations in
`ci-results/<index>/manifest.json`, JUnit results in `junit.xml`, and raw coverage
data in `.coverage` in the same directory. These are retained as CI artifacts.
The `test_coverage` aggregator verifies that the ten inventories form a complete,
disjoint partition of the selected tests before combining
coverage. It enforces the same 84% global minimum and the module floors from
`coverage-floors.toml` against the combined report. Per-shard coverage percentages
are not meaningful thresholds: each shard exercises only part of the suite.

To reproduce shard 1 locally, use:

```sh
mkdir -p ci-results/1
COVERAGE_FILE=ci-results/1/.coverage \
python -m pytest -p scripts.ci_shard tests -W error \
  -m "cpu or (neuron and not cuda)" \
  --ci-shard-index=1 --ci-shard-total=10 --ci-shard-output=ci-results/1 \
  --ci-shard-durations=scripts/ci_durations.json \
  --junitxml=ci-results/1/junit.xml --durations=25 \
  --cov=dendra --cov-branch --cov-config=.coveragerc.cpu --cov-report=
```

Repeat with shard indices 2 through 10, changing each output path to match its
index. Preserve the same checkout, dependencies, marker expression, and coverage
configuration, including the same duration profile, across all ten runs. Then
validate the inventories, combine raw coverage data, and generate the reports
with:

```sh
python scripts/combine_ci_coverage.py --shards 10 --input-dir ci-results --fail-under=84
```

The combined reports are written to `coverage.json`, `coverage.xml`, and
`htmlcov/` at the checkout root.

Coverage uses relative source paths so data collected in separate CI workspaces
can be combined. Combining the data preserves statement and branch coverage
across the entire lane; averaging shard percentages would not.

Sharding does not shorten or remove tests. Modules stay together to preserve
module-scoped fixture reuse. CI supplies the measured module-duration profile
in `scripts/ci_durations.json`: each module's estimate uses its measured time
per test multiplied by its currently selected case count. Modules are assigned
largest estimate first to the shard with the lowest estimated load. A module
absent from the profile uses the profile's total measured time divided by its
total measured test count as the fallback time per test. Omitting the optional
`--ci-shard-durations` argument balances selected case counts instead.

Shard manifests record the profile path, hash, and provenance alongside actual
test durations, so measurements from successful complete runs can inform later
profile updates. These estimates exclude installation and collection time.
Runner availability, hardware, compilation caches, and workload changes affect
wall-clock time; ten shards do not guarantee a particular speedup or maximum
job duration. The shorter development pass remains useful while iterating, but
the full CI lane remains mandatory.
