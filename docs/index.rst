Welcome to Dendra!
==================

``Dendra`` is a biophysical neuron & network simulator written in `PyTorch <https://github.com/pytorch/pytorch>`_, with an emphasis on effects of extracellular stimulation and event-based networks with synaptic delays. Its key features are:

- support for CPU and GPU
- automatic differentiation, facilitating gradient-based optimization of large numbers of parameters
- ``jit``-compilation (for speed and memory efficiency) via `torch.compile <https://pytorch.org/docs/stable/generated/torch.compile.html>`_
- symbolic specification of ODE and kinetic schemes for easy implementation of new mechanisms
- flexible extracellular stimulation with support for complex 3D fields
- fully differentiable event-based and continuous network simulation with synaptic delays
- a simple API, making it easy to use for beginners and experts alike
- built-in Hodgkin--Huxley and passive membrane mechanisms, plus symbolic APIs for implementing additional models
- a ``Materials`` interface for generic chemical reaction-diffusion simulations, including support for diffusion in intracellular and extracellular space


``Dendra`` is a research project and is still under development. If you have any questions, suggestions, or feedback, please use the issue tracker on the repository host available to you.

Getting started
---------------

``Dendra`` allows you to simulate the effects of stimulation on large numbers of biophysical neuron models on CPU or GPU:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   import dendra as dn
   from dendra.models.mod import hh
   from dendra.units import ms, nA

   # Stimulate one built-in Hodgkin--Huxley compartment.
   model = dn.SingleCompartment(
       N=1, C=1, celsius=6.3, cm=1.0, v_init=-65.0
   )
   model.diam.fill_(20.0)
   model.dx.fill_(20.0)
   model.insert(hh)
   model[..., 0].inject(
       dn.mono_rect(amp=0.1 * nA, delay=1.0 * ms, pw=1.0 * ms)
   )

   recorder = dn.callbacks.Recorder(["v"], node_indices=[0])
   dt = 0.01 * ms
   model.initialize()
   model.run(
       tstop=10.0 * ms,
       dt=dt,
       callbacks=[recorder],
       progressbar=False,
   )

   voltage = recorder.numpy("v")[:, 0, 0]
   time = dt * np.arange(len(voltage))
   plt.plot(time, voltage)
   plt.xlabel("Time (ms)")
   plt.ylabel("Membrane potential (mV)")
   plt.show()


Installation
------------

Dendra targets Python 3.11 or newer and PyTorch 2.12 or newer.

On Windows, run these commands inside WSL2 because the required NEURON package does not publish native Windows wheels on PyPI.

Until the first Dendra release is published on PyPI, install from a source checkout. From the repository root, run ``python -m pip install '.[solvers]'``. Use ``python -m pip install .`` if the optional native solvers are unavailable.

Once the distribution is published, a typical PyPI setup is:

1. Create and activate an isolated environment (optional): ``conda create -n dendra python=3.12 && conda activate dendra``.
2. If you need a particular CPU, CUDA, or ROCm build, install PyTorch first using its `installation selector <https://pytorch.org/get-started/locally/>`_.
3. Install Dendra with the recommended native CPU solvers: ``python -m pip install --only-binary=dendra-solvers 'dendra[solvers]'``.

If ``dendra-solvers`` cannot be installed, retry with ``python -m pip install dendra``. CPU unbranched cables can then use Dendra's built-in PyTorch solver; CPU block and tree methods require the optional package. GPU solvers are included with Dendra. See :ref:`installation` for wheel-only installation and supported platforms.

To build these docs locally without executing the notebooks, clone the source repository, install the documentation extras from its root (``python -m pip install '.[doc]'``), and run ``sphinx-build -W --keep-going -D nb_execution_mode=off -b html docs docs/_build/html``.

See :ref:`installation` for detailed guidance and optional extras (Jupyter and development tooling).


Feedback and Contributions
--------------------------

We welcome issues and merge or pull requests on the repository host available to you. External contributions will use GitHub when the public repository opens. When reporting a bug, include your OS, Python/PyTorch versions, install method, and a minimal reproducible script. For feature requests, please describe the workflow you are trying to support. The repository's ``CONTRIBUTING.md`` file describes setup, testing, and submission.

Contribution tips:

- Use a fresh branch and keep changes focused.
- Install development extras and CPU solvers (``python -m pip install --editable '.[dev,solvers]'``) for the full CPU test suite and coverage checks. Use ``.[dev]`` for development without native CPU solvers.
- Follow the existing style conventions; install both hook stages with ``pre-commit install`` and ``pre-commit install --hook-type commit-msg``.
- Documentation updates are appreciated—adding docstrings or short narrative sections to accompany new code is ideal.


Citation
--------

If you use `Dendra`, consider citing the `corresponding paper <https://www.nature.com/articles/s41467-024-51709-8>`_:

.. code-block:: console

    @article{hussain_highly_2024,
        title = {Highly efficient modeling and optimization of neural fiber responses to electrical stimulation},
        volume = {15},
        copyright = {2024 The Author(s)},
        issn = {2041-1723},
        url = {https://www.nature.com/articles/s41467-024-51709-8},
        doi = {10.1038/s41467-024-51709-8},
        language = {en},
        number = {1},
        urldate = {2024-09-01},
        journal = {Nature Communications},
        author = {Hussain, Minhaj A. and Grill, Warren M. and Pelot, Nicole A.},
        month = aug,
        year = {2024},
        note = {Publisher: Nature Publishing Group},
        keywords = {Machine learning, Peripheral nervous system, Biophysical models, Autonomic nervous system, Computational models},
        pages = {7597},
    }

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   installation
   upgrading
   units

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials

   basics
   advanced

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Miscellaneous

   misc

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Resources

   mechanisms
   slices
   dendra
   license
