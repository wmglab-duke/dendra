(installation)=
# Installation

> 💡 We recommend using a dedicated environment (e.g., `conda`) for Dendra.
> Dendra targets Python 3.11+; the examples below use Python 3.12.

```{important}
On Windows, we recommend using the Windows Subsystem for Linux (WSL2) for best compatibility and performance. 'Native' Windows is currently not supported for GPU simulations.
```

## Prerequisites

- Python 3.11 or newer
- PyTorch 2.7+ (install the CUDA 12.9+ wheels if you want GPU support)
- Git

## Quick start

1. Create and activate an environment (optional but recommended):

   ```sh
   conda create -n dendra python=3.12
   conda activate dendra
   ```

2. Install PyTorch (pick the command that matches your hardware):

   ```{important}
   If installing on Windows without WSL2, please follow PyTorch installation instructions [here](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) to ensure proper setup of the PyTorch Inductor backend (used by Dendra for JIT-compilation of models).
   ```

   ```{tip}
   We recommend installing the most recent stable version of PyTorch that supports your CUDA version. If you have an older GPU that is not compatible with the latest CUDA, you may need to install an older version of PyTorch that supports your CUDA version. See the [PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/) for more details.
   ```

   ```sh
   # e.g., GPU build (Pytorch 2.8.0, CUDA 12.9) - adjust the version and CUDA version as needed
   python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129

   # CPU-only build
   # python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
   ```

3. (Windows without WSL2) Install NEURON using the precompiled installer from https://neuron.yale.edu/neuron/download.

4. Clone the repository and install Dendra:

   ```sh
   git clone https://gitlab.oit.duke.edu/mah148/dendra.git
   cd dendra
   pip install .
   ```

## Optional extras

- Jupyter support: `pip install ".[jupyter]"`

- CPU implicit solvers: install the companion package [`dendra-solvers`](https://gitlab.oit.duke.edu/mah148/dendra-solvers) (required only for CPU implicit methods; GPU solvers are included by default). Also required to build the documentation.

- Documentation build dependencies: `pip install ".[doc]"` then `cd docs && make html`

- Development setup (editable install + lint/test tooling):

  ```sh
  pip install --editable ".[dev]"
  pre-commit install
  ```

- Library of models: install the companion package [`dendra-models`](https://gitlab.oit.duke.edu/mah148/dendra-models) for additional pre-defined neuron & network models.

## Verify the install

```sh
python - <<'PY'
import dendra as dn

print("Dendra import succeeded")
PY
```
