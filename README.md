<div align="center">
  <img src="docs/banner2.png">
</div>

***

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style-plastic&logo=PyTorch&logoColor=white)](https://pytorch.com)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Fast and scalable neural fiber simulator with extracellular field support. Implement and train high-throughput GPU-compatible models.

## ❗Requirements

### OS requirements
`axonml` has been tested on Windows 11 under WSL2 (Ubuntu 22.04) and Linux (AlmaLinux v9.3, binary-compatible with Red Hat Enterprise Linux).

### Python dependencies
`axonml` requires Python 3.12+ and PyTorch 2.7+ with GPU support (tested with PyTorch 2.7.0+ & CUDA 12.8).


## 🖥️ Installation

> [!TIP]
> We recommend using `conda` to manage your python environment. If you have `conda` installed, you may wish to set up a new environment: `conda create -n axonml python=3.12`. Be sure to activate your new environment (`conda activate axonml`) before following the installation instructions or running code.

1. Install PyTorch + CUDA 12.8.
```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu128
```

2. Clone this repository.

```bash
> git clone https://gitlab.oit.duke.edu/mah148/axonml.git
```

3. Install.

```bash
> cd axonml
> python -m pip install .
```
- To install in development mode:
    - `python -m pip install --editable '.[dev]'`

- If you want to build and run the documentation locally:
    - `python -m pip install '.[doc]'`

🥳 You're all set!

> [!NOTE]
> Installation of all dependencies should not take more time than a couple of minutes, depending on your internet speed. All dependencies (mainly PyTorch + CUDA libraries) require ~2GB of hard drive space.

> [!IMPORTANT]
> To use implicit euler when solving the voltage **on CPU**, install [axonml-solvers](https://gitlab.oit.duke.edu/mah148/axonml-solvers). GPU implementation is available by default.

## 🗄️ Pre-implemented models

Cell models are available at https://gitlab.oit.duke.edu/mah148/axonml-models.

## 📜 License
The copyrights of this software are owned by Duke University. As such, it is offered under a custom license (see LICENSE.md) whereby:

1. DUKE grants YOU a royalty-free, non-transferable, non-exclusive, worldwide license under its copyright to use, reproduce, modify, publicly display, and perform the PROGRAM solely for non-commercial research and/or academic testing purposes.

2. In order to obtain any further license rights, including the right to use the PROGRAM, any modifications or derivatives made by YOU, and/or PATENT RIGHTS for commercial purposes, (including using modifications as part of an industrially sponsored research project), YOU must contact DUKE’s Office for Translation and Commercialization (Digital Innovations Team) about additional commercial license agreements.

Please note that this software is distributed AS IS, WITHOUT ANY WARRANTY; and without the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

[^1]: Storn, Rainer, and Kenneth Price. 1997. “Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces.” Journal of Global Optimization 11 (4): 341–59. https://doi.org/10.1023/A:1008202821328.

[^2]: McIntyre, Cameron C., Andrew G. Richardson, and Warren M. Grill. 2002. “Modeling the Excitability of Mammalian Nerve Fibers: Influence of Afterpotentials on the Recovery Cycle.” Journal of Neurophysiology 87 (2): 995–1006. https://doi.org/10.1152/jn.00353.2001.
