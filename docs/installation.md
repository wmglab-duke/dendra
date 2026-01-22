(installation)=
# Installation

> 💡 We recommend using a dedicated environment (e.g., `conda`) for AxonML.
> AxonML targets Python 3.11+; the examples below use Python 3.12.

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
   conda create -n axonml python=3.12
   conda activate axonml
   ```

2. Install PyTorch (pick the command that matches your hardware):

   ```{important}
   PyTorch 2.9+ is supported but exhibits some stochastic performance regressions on small-batch CPU simulation. If this is your use case (relatively small numbers of individual fiber simulations), we recommend PyTorch 2.8.0 for best performance.
   ```

   ```{important}
   If installing on Windows without WSL2, please follow PyTorch installation instructions [here](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html) to ensure proper setup of the PyTorch Inductor backend (used by AxonML for JIT-compilation of models).

   ```sh
   # e.g., GPU build (Pytorch 2.8.0, CUDA 12.9)
   python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129

   # CPU-only build
   # python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
   ```

3. (Windows without WSL2) Install NEURON using the precompiled installer from https://neuron.yale.edu/neuron/download.

4. Clone the repository and install AxonML:

   ```sh
   git clone https://gitlab.oit.duke.edu/mah148/axonml.git
   cd axonml
   python -m pip install .
   ```

## Optional extras

- Jupyter support: `python -m pip install '.[jupyter]'`

- CPU implicit solvers: install the companion package [`axonml-solvers`](https://gitlab.oit.duke.edu/mah148/axonml-solvers) (required only for CPU implicit methods; GPU solvers are included by default). Also required to build the documentation.

- Documentation build dependencies: `python -m pip install '.[doc]'` then `cd docs && make html`

- Development setup (editable install + lint/test tooling):

  ```sh
  python -m pip install --editable '.[dev]'
  python -m pre-commit install
  ```

- Library of models: install the companion package [`axonml-models`](https://gitlab.oit.duke.edu/mah148/axonml-models) for additional pre-defined neuron & network models.

## Verify the install

```sh
python - <<'PY'
import axonml as ax

print("AxonML import succeeded")
PY
```
