(installation)=
# Installation

```{note}
On Windows, we recommend Windows Subsystem for Linux (WSL2). The shell commands
below use Linux syntax.
```

## Requirements

Use Python 3.11 or newer with pip. A dedicated environment keeps Dendra's
dependencies separate from other projects.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

With Conda, use
`conda create -n dendra python=3.12 && conda activate dendra`.
Source and development installations also require Git.

(hardware-support)=
## Hardware support

| Hardware | PyTorch device | Solver setup |
| --- | --- | --- |
| CPU | `cpu` | The base package supports unbranched cables. Use the `solvers` extra for native block and tree solvers. |
| GPU | `cuda` | Dendra includes Triton solvers tested with NVIDIA CUDA. See {ref}`CUDA deployment <gpu-deployment>`. |
| Apple Silicon | `mps` | Use `torch.float32`. The `solvers` extra enables native Metal solvers. See {ref}`apple-silicon`. |

AMD GPU compatibility with ROCm is unverified. PyTorch's ROCm build also uses
[`device="cuda"`](https://docs.pytorch.org/docs/stable/notes/hip.html), and
[Triton can target AMD GPUs](https://github.com/triton-lang/triton#compatibility).
The optional native NetCon extension and `dendra doctor --require-cuda` target
NVIDIA CUDA.

(choose-pytorch-build)=
## Choose a PyTorch build

Skip this step to let pip install PyTorch automatically. For a specific CPU,
CUDA, or ROCm build, install PyTorch 2.14 or newer using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).

For a CPU-only build on Linux or WSL2, use PyTorch's CPU wheel index.

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Use the PyTorch index only for the PyTorch installation.

## Install from PyPI

Install Dendra with the recommended native CPU and Apple Metal solvers.

```sh
python -m pip install --only-binary=dendra-solvers "dendra[solvers]"
```

If pip reports that a compatible solver wheel is unavailable, install the
base package. Pip installs the required PyTorch and NEURON dependencies in
either case.

```sh
python -m pip install dendra
```

## Verify the install

```sh
python -m pip check
python -m dendra doctor
```

The doctor reports the installed Dendra, Python and PyTorch versions. It
also reports CUDA devices, toolchain discovery, and the optional NetCon
extension's build and cache status. For CUDA, see the
{ref}`deployment checks <gpu-deployment>`. See the checks below for the
{ref}`native CPU extension <native-implicit-solvers>` and {ref}`apple-silicon`.

(native-implicit-solvers)=
## Native CPU and MPS implicit solvers

The optional `solvers` extra requires
[`dendra-solvers`](https://pypi.org/project/dendra-solvers/) 0.5.1 or newer.
CPU block and tree methods use this package. The base package uses Dendra's
PyTorch parallel cyclic reduction (PCR) solver for unbranched CPU cables.
For Metal solver requirements and device selection, see {ref}`apple-silicon`.

To add native solvers to an existing Dendra environment, install a prebuilt
wheel.

```sh
python -m pip install --only-binary=dendra-solvers "dendra-solvers>=0.5.1"
```

Prebuilt wheels are available for Linux x86-64 and ARM64 with glibc 2.24 or
newer, Windows x86-64, and macOS 14 or newer on Apple Silicon. Inside WSL2, pip
selects the Linux wheel.

If a compatible wheel is unavailable, you can omit `--only-binary=dendra-solvers`
to attempt a local build with a C++ toolchain. See the
[`dendra-solvers` package page](https://pypi.org/project/dendra-solvers/) for
source-build instructions.

Check that the installed CPU extension loads.

```sh
python - <<'PY'
from importlib.metadata import version
import dendra_solvers._ext as solver_ext

print("dendra-solvers:", version("dendra-solvers"))
print("CPU solver extension:", solver_ext.__file__)
PY
```

(apple-silicon)=
## Apple Silicon (MPS)

On Apple Silicon, PyTorch's MPS backend runs simulations on the Apple GPU.
Select `device="mps"` and `dtype=torch.float32` when constructing the model.

```python
import torch
import dendra as dn

model = dn.SingleCompartment(N=1, C=1, device="mps", dtype=torch.float32)
```

The native Metal backend supplies Thomas and PCR cable solvers and scalar,
block, and packed DHS tree solvers. It requires macOS 14 or newer, an
available MPS device, and the `solvers` extra. Cable solvers can fall back to
PyTorch on MPS. DHS on MPS requires the native backend. For scalar cables on
MPS, choose Thomas or PCR.

Check whether PyTorch can use MPS and whether the native Metal solver backend
is available.

```sh
python -c "import torch; print('PyTorch MPS:', torch.backends.mps.is_available())"
python -c "import dendra_solvers; print('Native MPS:', dendra_solvers.is_mps_available())"
```

See {doc}`testing` for CPU and MPS solver integration tests.

## Install from a source checkout

To install from source, clone the public repository and install from its root.

```sh
git clone https://github.com/wmglab-duke/dendra.git
cd dendra
python -m pip install ".[solvers]"
```

Use `python -m pip install .` if you need the base package. For an editable
development installation with lint and test dependencies, follow the
[contribution guide](https://github.com/wmglab-duke/dendra/blob/main/CONTRIBUTING.md).

## Optional extras

- For interactive plots, see {ref}`interactive Jupyter setup <interactive-jupyter>`.
- For a prose-only documentation preview from a source checkout, install
  `python -m pip install ".[doc]"` and run
  `sphinx-build -W --keep-going -D nb_execution_mode=off -b html docs docs/_build/html`.
  The repository's `docs/README.md` gives the full workflow with companion
  models and notebook execution.

(interactive-jupyter)=
### Interactive Jupyter figures

The `ipympl` package supplies a backend for the notebook kernel and a prebuilt
`jupyter-matplotlib` extension for the Jupyter server. If the server and
kernel share an environment, use these steps.

1. Install the extra in that environment.

   ```sh
   python -m pip install "dendra[jupyter]"
   ```

2. Save your notebooks and restart the Jupyter server from that environment.
   The server discovers the extension when it starts.

   ```sh
   python -m jupyter lab
   ```

3. Refresh the page, start a fresh kernel, and select the backend before
   creating figures.

   ```ipython
   %matplotlib widget
   ```

If the server and kernel use separate environments, install Dendra and ipympl
in the kernel environment. Install ipympl in the server environment, using the
same version where possible, then restart both. A managed JupyterHub may
require an administrator to install the server extension. JupyterLab 3 and 4
and Notebook 7 can use the prebuilt extension directly.

To check the server, run `jupyter labextension list` with the executable that
launches Jupyter. Both `jupyter-matplotlib` and
`@jupyter-widgets/jupyterlab-manager` should show `enabled` and `OK`. In the
kernel, run `import sys, ipympl; print(sys.executable, ipympl.__version__)` to
identify the Python environment and backend version.

The error `Failed to load model class 'MPLCanvasModel' from module
'jupyter-matplotlib'` means the browser could not load a compatible extension.
Check the server environment, restart the server, and open a fresh page.

## Upgrade

To upgrade Dendra with the solver extra, run the following command.

```sh
python -m pip install --upgrade --only-binary=dendra-solvers "dendra[solvers]"
```

### Compiler compatibility

On affected PyTorch builds, Dendra applies a compatibility repair when it
creates the first compiled program. The repair avoids redundant compilation
retries and retains scalar guards. The first repaired compilation creates
fresh cache artifacts that later processes can reuse. The repair preserves
your compiler settings and any existing cache tag.

### Optional TorchInductor cache isolation

Dendra uses PyTorch's cache defaults unless you enable its TorchInductor
configuration. To use an isolated cache per process, set this environment
variable before starting Python.

```sh
export DENDRA_CONFIGURE_TORCHINDUCTOR_CACHE=1
python your_simulation.py
```

The option also enables Dendra's precompiled-header, compile-thread and CPU
instruction-set settings. Set `DENDRA_INDUCTOR_CACHE_POLICY=shared` to reuse a
shared cache, or `off` to disable configuration. Import Dendra before PyTorch
when using these settings.

```python
import dendra as dn
import torch
```
