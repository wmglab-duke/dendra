(installation)=
# Installation

> 💡 We recommend using a dedicated environment (e.g., `conda`) for Dendra.
> Dendra targets Python 3.11+; the examples below use Python 3.12.

```{important}
On Windows, use the Windows Subsystem for Linux (WSL2) and follow the Linux
commands below. Dendra's required NEURON dependency does not publish native
Windows wheels on PyPI, so the standard Dendra PyPI installation is not
supported in native PowerShell or Command Prompt.
```

## Requirements

- Python 3.11 or newer
- Pip

Dendra requires PyTorch 2.12 or newer and NEURON. Both are declared package
dependencies and are installed automatically. Git is needed only for source
and development installations.

## Install from a source checkout

Until the first Dendra release is published on PyPI, install from your source
checkout. From the repository root, run:

```sh
python -m pip install --upgrade pip
python -m pip install ".[solvers]"
```

Use `python -m pip install .` if the optional native solvers are unavailable.
For development, use an editable installation as described in the repository's
`CONTRIBUTING.md` file.

## Install from PyPI

The commands in this section apply once the Dendra distribution has been
published on PyPI.

1. Create and activate an environment (optional but recommended):

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   ```

   Conda users can instead run
   `conda create -n dendra python=3.12 && conda activate dendra`.

2. Install Dendra with the recommended native CPU solvers:

   ```sh
   python -m pip install --upgrade pip
   python -m pip install --only-binary=dendra-solvers "dendra[solvers]"
   ```

   The `solvers` extra installs
   [`dendra-solvers`](https://pypi.org/project/dendra-solvers/), which provides
   native CPU solvers for block and tree models.

3. If no compatible `dendra-solvers` distribution is available for your
   platform, install the base package instead:

   ```sh
   python -m pip install dendra
   ```

   The base package retains Dendra's PyTorch CPU solver for unbranched cables
   and all GPU solvers. CPU block and tree methods require `dendra-solvers`.

## Choose a PyTorch build

Pip installs a compatible PyTorch release automatically. If you need a
particular CPU, CUDA, or ROCm build, install it before Dendra using the command
from the [PyTorch installation selector](https://pytorch.org/get-started/locally/).
Dendra requires PyTorch 2.12 or newer.

For example, a CPU-only installation on Linux or WSL2 uses PyTorch's CPU wheel
index:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

After installing the selected build, install Dendra without carrying the
PyTorch index into that command:

```sh
python -m pip install --only-binary=dendra-solvers "dendra[solvers]"
```

On Windows hosts, run this inside WSL2.

## CPU implicit solvers

The quick start includes [`dendra-solvers`](https://pypi.org/project/dendra-solvers/)
through the optional `solvers` extra.
The base installation (`python -m pip install dendra`) does not download or
build it, so an unavailable solver wheel or failed solver build does not
prevent you from installing Dendra.

Without this package, CPU unbranched cable integration falls back to Dendra's
PyTorch parallel cyclic reduction (PCR) solver. CPU block and tree methods
require `dendra-solvers`. GPU solvers are included with Dendra and do not
depend on this package.

To add a prebuilt native CPU solver to an existing installation:

```sh
python -m pip install --only-binary=dendra-solvers "dendra[solvers]"
```

Or add the package to an existing Dendra installation, allowing only prebuilt
wheels:

```sh
python -m pip install --only-binary=dendra-solvers "dendra-solvers>=0.3.1"
```

The wheel-only command fails if there is no compatible wheel, leaving your
existing Dendra installation usable. To attempt a local source build when a
wheel is unavailable, omit `--only-binary=dendra-solvers`; this requires a
working C++ build environment and can still fail. Use the base installation in
that case.

Prebuilt solver wheels are available for:

- Linux x86-64 and ARM64 on glibc 2.24 or newer.
- Windows x86-64.
- macOS 14 or newer on Apple Silicon.

When a matching wheel is available, pip selects it automatically, so installing
the CPU solvers does not require a C++ compiler. For other platforms or
custom PyTorch builds, see the source-build instructions on the
[`dendra-solvers` package page](https://pypi.org/project/dendra-solvers/).
The standalone Windows solver wheel supports CPU kernels, but it does not
remove Dendra's WSL2 requirement because NEURON remains a required dependency
of the full Dendra package.

## Optional extras

- Interactive Jupyter plots: `python -m pip install "dendra[jupyter]"`. See the
  {ref}`interactive Jupyter setup <interactive-jupyter>` below.

- For a fast prose-only documentation preview, install the documentation dependencies with
  `python -m pip install ".[doc]"`, then run
  `sphinx-build -W --keep-going -D nb_execution_mode=off -b html docs docs/_build/html`.
  The repository's `docs/README.md` gives the full CI-equivalent workflow,
  which installs the companion models and executes every notebook.

- For an editable development installation with lint, test, and CPU-solver
  dependencies, follow the repository's `CONTRIBUTING.md` file.

## Upgrade

Upgrade Dendra while retaining the recommended solver extra with:

```sh
python -m pip install --upgrade --only-binary=dendra-solvers "dendra[solvers]"
```

### Optional TorchInductor cache isolation

Importing Dendra leaves TorchInductor's cache behavior at PyTorch's defaults.
Dendra's cache-directory, precompiled-header, compile-thread, and CPU ISA cache
overrides are all opt-in. To enable them with per-process cache isolation, set
the following environment variable before starting Python:

```sh
export DENDRA_CONFIGURE_TORCHINDUCTOR_CACHE=1
python your_simulation.py
```

With this option enabled, the default cache policy is `process`. Set
`DENDRA_INDUCTOR_CACHE_POLICY=shared` to retain a shared cache, or `off` to
disable configuration even when the import gate is enabled. Dendra must be
imported before PyTorch so the cache settings take effect reliably:

```python
import dendra as dn
import torch
```

(interactive-jupyter)=
### Interactive Jupyter figures

The optional `ipympl` package has two cooperating parts: a Python backend in
the notebook kernel and a prebuilt `jupyter-matplotlib` extension in the
environment running the Jupyter server.

When the server and kernel use the same environment:

1. Install the extra in that environment:

   ```sh
   python -m pip install "dendra[jupyter]"
   ```

2. Save your notebooks and stop the **entire Jupyter server**. Restarting only
   the kernel is insufficient after the first installation because the
   already-running Lab frontend has not discovered the new extension. Relaunch
   Jupyter from the activated environment, for example:

   ```sh
   python -m jupyter lab
   ```

3. Open or hard-refresh the Lab page, start a fresh kernel, and select the
   backend before creating figures:

   ```ipython
   %matplotlib widget
   ```

If the server and kernel use **separate environments**, install Dendra and
ipympl in the kernel environment, then install a compatible—preferably the
same—ipympl release in the environment that launches the per-user Jupyter
server. Restart both. On a managed JupyterHub, server-side installation may
require an administrator. For JupyterLab 3/4 and Notebook 7, ipympl ships a
prebuilt extension: do not run `jupyter lab build` or manually install a Lab
extension.

To verify the server side, run `jupyter labextension list` with the executable
that launches Jupyter. It should report both `jupyter-matplotlib` and
`@jupyter-widgets/jupyterlab-manager` as `enabled` and `OK`. In the kernel,
`import sys, ipympl; print(sys.executable, ipympl.__version__)` identifies the
Python environment and backend version.

The browser error `Failed to load model class 'MPLCanvasModel' from module
'jupyter-matplotlib'` means the kernel backend is active but the current Lab
page did not load a compatible frontend module. In a shared environment,
restart the whole server and open a fresh page. In a split environment, verify
the server-side installation before restarting.

## Verify the install

First ask pip to check the complete dependency set and run Dendra's deployment
diagnostics:

```sh
python -m pip check
python -m dendra doctor
```

The doctor command reports the PyTorch device, CUDA readiness, compiler and
toolkit discovery, and Dendra's optional NetCon CUDA extension. For a GPU
deployment that must provide CUDA, run
`python -m dendra doctor --require-cuda`.

You can also verify the import and published version directly:

```sh
python - <<'PY'
from importlib.metadata import version
import dendra as dn

print("Dendra:", version("dendra"), dn.__file__)
PY
```

If you installed the optional CPU solvers, verify that their native extension
loads too:

```sh
python - <<'PY'
from importlib.metadata import version
import dendra_solvers._ext as solver_ext

print("dendra-solvers:", version("dendra-solvers"))
print("CPU solver extension:", solver_ext.__file__)
PY
```
