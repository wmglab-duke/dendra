<div align="center">
  <img src="https://raw.githubusercontent.com/wmglab-duke/dendra/main/docs/_static/logo-light.png" alt="Dendra">
</div>

***

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style-plastic&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-Ruff-D7FF64.svg?logo=ruff&logoColor=261230)](https://docs.astral.sh/ruff/)

Dendra is a differentiable biophysical neuron and network simulator built on
PyTorch. It supports CPU and GPU execution, extracellular fields, event-based
networks, and macroscopic descriptors for gradient-based model training.

## Documentation

The complete documentation is published at
[wmglab-duke.github.io/dendra](https://wmglab-duke.github.io/dendra/). Start
with the [installation guide](https://wmglab-duke.github.io/dendra/installation.html),
then see the tutorials and the
[unit conventions](https://wmglab-duke.github.io/dendra/units.html).

## Installation

Dendra requires Python 3.11 or newer and PyTorch 2.12 or newer. On Windows,
install inside WSL2 because the required NEURON dependency does not publish
native Windows wheels on PyPI.

Install Dendra with the recommended native CPU solvers:

```bash
python -m pip install --upgrade pip
python -m pip install --only-binary=dendra-solvers "dendra[solvers]"
```

The `solvers` extra installs
[`dendra-solvers`](https://pypi.org/project/dendra-solvers/), which provides
native CPU solvers for block and tree models. If a compatible solver wheel is
unavailable, install the base package:

```bash
python -m pip install dendra
```

The base package retains Dendra's PyTorch CPU solver for unbranched cables and
all GPU solvers. To select a particular CPU, CUDA, or ROCm build, install
PyTorch first using its
[installation selector](https://pytorch.org/get-started/locally/), then install
Dendra.

For interactive Matplotlib figures in Jupyter, install the Jupyter extra:

```bash
python -m pip install "dendra[jupyter]"
```

Verify the installed dependencies and inspect the deployment environment:

```bash
python -m pip check
python -m dendra doctor
```

The [installation guide](https://wmglab-duke.github.io/dendra/installation.html)
explains solver-wheel support, source installations, upgrades, and Jupyter
setup. For CUDA deployment checks, see the
[GPU deployment guide](https://wmglab-duke.github.io/dendra/gpu-deployment.html).

## Development

The [contribution guide](https://github.com/wmglab-duke/dendra/blob/main/CONTRIBUTING.md)
describes development setup and the GitHub pull-request workflow. The
[testing and code coverage guide](https://wmglab-duke.github.io/dendra/testing.html)
covers the CPU, NEURON, CUDA, sanitizer, and coverage lanes.

## Citation

A citation for Dendra and its accompanying manuscript will be added when it is
available. Please also consider citing the work where geometric and topological
surrogates and gradient-based design of selective neurostimulation are
discussed in detail:

Minhaj A. Hussain, Warren M. Grill, Nicole A. Pelot. "Highly efficient
modeling and optimization of neural fiber responses to electrical
stimulation." *Nature Communications.* 2024.
[doi:10.1038/s41467-024-51709-8](https://doi.org/10.1038/s41467-024-51709-8)

```bibtex
@article{hussain_highly_2024,
    title = {Highly efficient modeling and optimization of neural fiber responses to electrical stimulation},
    doi = {10.1038/s41467-024-51709-8},
    journal = {Nature Communications},
    author = {Hussain, Minhaj A. and Grill, Warren M. and Pelot, Nicole A.},
    year = {2024}
}
```

## License

Dendra is distributed under Duke University's custom license for non-commercial
research and academic testing. Commercial use, including industrially
sponsored research, requires a separate agreement with Duke's Office for
Translation and Commercialization. The
[license](https://github.com/wmglab-duke/dendra/blob/main/LICENSE.md) governs
all use; downloading, cloning, or forking the software constitutes acceptance
of those terms.
