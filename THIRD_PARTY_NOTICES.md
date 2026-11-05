# Third-party notices

Dendra's original code is covered by the [Duke University agreement](LICENSE.md).
The third-party portions identified below retain their upstream license terms.

## PyTorch

Parts of the scan compatibility adapters in `dendra/func/_scan_compat.py` are
adapted from [PyTorch's higher-order operator utilities](https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/torch/_higher_order_ops/utils.py).

These upstream portions are licensed under BSD-3-Clause. The complete PyTorch
copyright notices, license conditions and disclaimer are included in
[`LICENSES/PyTorch.txt`](LICENSES/PyTorch.txt).

## NEURON

`tests/simtests/111200A.asc` is reproduced unchanged from
[NEURON's morphology import example](https://github.com/neuronsimulator/nrn/blob/9.0.1/docs/courses/code/111200A.asc).
Its BSD-3-Clause copyright notice, license conditions and disclaimer are included
in [`LICENSES/NEURON.txt`](LICENSES/NEURON.txt).

## Blue Brain morphology

`docs/basics/L23_PC_cADpyr229_1.asc` is a Blue Brain Project L2/3 pyramidal
morphology (Copyright © 2015 EPFL-BBP), from the
[original cell package](https://bbp.epfl.ch/nmc-portal/assets/documents/static/downloads-zip/L23_PC_cADpyr229_1.zip)
described by [Markram et al. (2015)](https://doi.org/10.1016/j.cell.2015.09.029).
The file was renamed, attribution comments were added, and two trailing blank
lines were removed; its geometry is unchanged.

The morphology retains its
[CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
The [upstream notice](LICENSES/BlueBrain.txt) and
[complete license](LICENSES/CC-BY-NC-SA-4.0.txt) are included with Dendra.
This license applies to the morphology file.

Separately installed dependencies carry their own license files and notices.
