(installation)=
# 🖥️ Installation

> 💡
> We recommend using `conda` to manage your python environment. If you have `conda` installed, you may wish to set up a new environment: `conda create -n axonml python=3.11`. Be sure to activate your new environment (`conda activate axonml`) before following the installation instructions or running code.

1. Clone this repository recursively.

```sh
git clone --recursive https://gitlab.oit.duke.edu/mah148/axonml.git
```

2. Install.

```sh
cd axonml
python -m pip install .
```

- You can also install with jupyter support:
    - `python -m pip install '.[jupyter]'`