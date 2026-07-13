<div align="center">
  <img src="docs/_static/logo-v2.png">
</div>

***

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style-plastic&logo=PyTorch&logoColor=white)](https://pytorch.com)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage](https://gitlab.oit.duke.edu/mah148/dendra/badges/main/coverage.svg?job=test)](https://gitlab.oit.duke.edu/mah148/dendra/-/pipelines?ref=main)

Fast, scalable, versatile, and differentiable neural simulator with support for extracellular fields. Useful to implement and train high-throughput GPU-compatible models.

## Documentation
Full documentation is available at [https://mah148.pages.oit.duke.edu/dendra](https://mah148.pages.oit.duke.edu/dendra).

## Requirements

### OS requirements
`dendra` has been tested on Windows 11 under WSL2 (Ubuntu 22.04), Linux (AlmaLinux v9.3, binary-compatible with Red Hat Enterprise Linux), and macOS (Tahoe 26.3).

### Python dependencies
`dendra` requires Python 3.11+, PyTorch 2.8+, and NEURON. For GPU support, CUDA 12.9+ is required for best performance. **We recommend installing the most recent stable version of PyTorch that supports your CUDA version**. If you have an older GPU that is not compatible with the latest CUDA, you may need to install an older version of PyTorch that supports your CUDA version. See the [PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/) for more details.

## 🖥️ Installation

> [!TIP]
> We recommend using `conda` to manage your python environment. If you have `conda` installed, you may wish to set up a new environment: `conda create -n dendra python=3.12`. Be sure to activate your new environment (`conda activate dendra`) before following the installation instructions or running code.

1. (Optional) Install your preferred version of PyTorch. If you do not have a GPU or do not need GPU support, we recommend you install the CPU-only version of PyTorch to avoid installing unnecessary CUDA dependencies. e.g., for PyTorch 2.8.0:
```bash
> pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129  # GPU version, CUDA 12.9 specified - adjust the version and CUDA version as needed
> pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu    # CPU-only version
```

2. Clone this repository.

```bash
> git clone https://gitlab.oit.duke.edu/mah148/dendra.git
```

3. Install.

```bash
> cd dendra
> pip install .
```

- If you want to build and run the documentation locally:
    - `pip install ".[doc]"`

### ⚙️ Installing for development
- Install `--editable` with dev dependencies & install `pre-commit`:
    - `pip install --editable ".[dev]"`
    - `pre-commit install`

## ✅ Testing and code coverage

The development dependencies include `pytest` and `pytest-cov`. Run the complete test suite from the repository root with:

```bash
python -m pytest tests
```

Tests are classified into execution lanes. The required CI lane combines the portable CPU suite with every non-CUDA NEURON integration and reference test:

```bash
python -m pytest tests -W error -m "cpu or (neuron and not cuda)"
```

This expression runs each selected test once: `cpu` covers tests that require neither CUDA nor NEURON, while `neuron and not cuda` adds the required simulator-backed CPU tests without pulling GPU comparisons into the lane. CUDA and Triton-dependent cases remain separate and can be selected with `-m cuda`; `-m "neuron and cuda"` narrows that lane to NEURON comparisons on CUDA. Longer comparisons also carry `slow`, and randomized/property-generated cases carry `stochastic`. Markers are strict, so misspelled or undeclared markers fail during collection.

On a CUDA/Triton worker, run the accelerator lane independently:

```bash
python -c "import torch, triton; assert torch.cuda.is_available()"
python -m pytest tests -W error -m cuda
```

Keep accelerator coverage artifacts separate from the required CPU/NEURON percentage. Kernel correctness is enforced primarily through dense numerical oracles, gradient checks, boundary-shape contracts, and backend-equivalence tests.

To measure both statement and branch coverage, print uncovered line numbers in the terminal, and generate a browsable HTML report:

```bash
python -m pytest tests -W error -m "cpu or (neuron and not cuda)" --cov=dendra --cov-branch --cov-report=term-missing:skip-covered --cov-report=json:coverage.json --cov-report=html
```

The `TOTAL` row is the overall coverage result. Open `htmlcov/index.html` to inspect coverage by module and identify untested lines and branches. The required lane installs and exercises NEURON; CUDA and Triton remain outside this coverage measurement, and results can still vary when optional CPU solvers are unavailable.

GitLab CI runs this same non-CUDA branch-coverage measurement for every pipeline, enforces a ratcheted global minimum, and retains JSON, browsable HTML, and Cobertura reports. Critical modules also have individual floors configured in `pyproject.toml`; validate them locally after generating `coverage.json` with:

```bash
python scripts/check_coverage_floors.py coverage.json
```


🥳 You're all set!

> [!NOTE]
> Installation of all dependencies should not take more time than a couple of minutes, depending on your internet speed. All dependencies (mainly PyTorch + CUDA libraries) require ~2GB of hard drive space.

> [!IMPORTANT]
> To enable implicit methods for solving $V_m$ **on CPU**, install [dendra-solvers](https://gitlab.oit.duke.edu/mah148/dendra-solvers). GPU implementations of all solvers are available by default.

## 🗄️ Pre-implemented models

Cell & network models are available at https://gitlab.oit.duke.edu/mah148/dendra-models.


## 🔍 Citation

If you use Dendra, please cite...(paper forthcoming).

Please also consider citing the work where geometric / topological surrogates and gradient-based design of selective neurostimulation are discussed in detail:

Minhaj A. Hussain, Warren M. Grill, Nicole A. Pelot. "Highly efficient modeling and optimization of neural fiber responses to electrical stimulation." *Nature Communications.* 2024. [(nature.com)](https://www.nature.com/articles/s41467-024-51709-8)

```
@article{hussain_highly_2024,
    title = {Highly efficient modeling and optimization of neural fiber responses to electrical stimulation},
    doi = {10.1038/s41467-024-51709-8},
    journal = {Nature Communications},
    author = {Hussain, Minhaj A. and Grill, Warren M. and Pelot, Nicole A.},
    year = {2024}
}
```


## 📜 License
The copyrights of this software are owned by Duke University. As such, it is offered under a custom license (see LICENSE.md) whereby:

1. DUKE grants YOU a royalty-free, non-transferable, non-exclusive, worldwide license under its copyright to use, reproduce, modify, publicly display, and perform the PROGRAM solely for non-commercial research and/or academic testing purposes.

2. In order to obtain any further license rights, including the right to use the PROGRAM, any modifications or derivatives made by YOU, and/or PATENT RIGHTS for commercial purposes, (including using modifications as part of an industrially sponsored research project), YOU must contact DUKE’s Office for Translation and Commercialization (Digital Innovations Team) about additional commercial license agreements.

Please note that this software is distributed AS IS, WITHOUT ANY WARRANTY; and without the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
