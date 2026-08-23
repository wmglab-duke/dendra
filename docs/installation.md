(installation)=
# Installation

> 💡 We recommend using a dedicated environment (e.g., `conda`) for Dendra.
> Dendra targets Python 3.11+; the examples below use Python 3.12.

```{important}
On Windows, we recommend using the Windows Subsystem for Linux (WSL2) for best compatibility and performance. 'Native' Windows is currently not supported for GPU simulations.
```

## Prerequisites

- Python 3.11 or newer
- PyTorch 2.8+
- Git

## Quick start

1. Create and activate an environment (optional but recommended):

   ```sh
   conda create -n dendra python=3.12
   conda activate dendra
   ```

2. (Optional) Install your preferred version of PyTorch:

   ```{important}
   If installing on Windows without WSL2, please follow PyTorch installation instructions [here](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) to ensure proper setup of the PyTorch Inductor backend (used when Dendra JIT compilation is enabled).
   ```

   ```{tip}
   We recommend installing the most recent stable version of PyTorch that supports your CUDA version. If you have an older GPU that is not compatible with the latest CUDA, you may need to install an older version of PyTorch that supports your CUDA version. See the [PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/) for more details.
   ```

   ```sh
   # e.g., GPU build (Pytorch 2.8.0, CUDA 12.9) - adjust the version and CUDA version as needed
   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129

   # CPU-only build
   # pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
   ```

3. (Windows without WSL2) Install NEURON using the precompiled installer from https://neuron.yale.edu/neuron/download.

4. Clone the repository and install Dendra:

   ```sh
   git clone https://gitlab.oit.duke.edu/mah148/dendra.git
   cd dendra
   pip install .
   ```

## Optional extras

- Interactive Jupyter plots: `pip install ".[jupyter]"`. See the
  {ref}`interactive Jupyter setup <interactive-jupyter>` below.

- CPU implicit solvers: install the companion package [`dendra-solvers`](https://gitlab.oit.duke.edu/mah148/dendra-solvers) (required only for CPU implicit methods; GPU solvers are included by default). Also required to build the documentation.

- Documentation build dependencies: `pip install ".[doc]"` then `cd docs && make html`

- Development setup (editable install + lint/test tooling):

  ```sh
  pip install --editable ".[dev]"
  pre-commit install
  ```

- Library of models: install the companion package [`dendra-models`](https://gitlab.oit.duke.edu/mah148/dendra-models) for additional pre-defined neuron & network models.

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
   pip install ".[jupyter]"
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

```sh
python - <<'PY'
import dendra as dn

print("Dendra import succeeded")
PY
```
