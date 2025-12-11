(installation)=
# Installation

> 💡 We recommend using a dedicated environment (e.g., `conda`) for AxonML.
> AxonML targets Python 3.11+; the examples below use Python 3.12.

## Prerequisites

- Python 3.11 or newer
- PyTorch 2.7+ (install the CUDA 12.9 wheels if you want GPU support)
- Git

## Quick start

1. Create and activate an environment (optional but recommended):

   ```sh
   conda create -n axonml python=3.12
   conda activate axonml
   ```

2. Install PyTorch (pick the command that matches your hardware):

   ```sh
   # GPU build (CUDA 12.9)
   python -m pip install torch --index-url https://download.pytorch.org/whl/cu129

   # CPU-only build
   # python -m pip install torch
   ```

3. Clone the repository and install AxonML:

   ```sh
   git clone https://gitlab.oit.duke.edu/mah148/axonml.git
   cd axonml
   python -m pip install .
   ```

## Optional extras

- Jupyter support: `python -m pip install '.[jupyter]'`

- CPU implicit solvers: install the companion package `axonml-solvers` (required only for CPU implicit methods; GPU solvers are included by default). Also required to build the documentation.

- Documentation build dependencies: `python -m pip install '.[doc]'` then `cd docs && make html`

- Development setup (editable install + lint/test tooling):

  ```sh
  python -m pip install --editable '.[dev]'
  python -m pre-commit install
  ```

- Library of models: install the companion package `axonml-models` for additional pre-defined neuron & network models.

## Verify the install

```sh
python - <<'PY'
import axonml as ax

print("AxonML import succeeded")
print("Available base types:", ax.Axon, ax.Unmyelinated, ax.Myelinated)
PY
```
