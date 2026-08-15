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
- implementations of a range of popular biophysical models, including Hodgkin-Huxley, Tigerholm, MRG, and more (via `Dendra Models <https://gitlab.oit.duke.edu/mah148/dendra-models>`_)
- a ``Materials`` interface for generic chemical reaction-diffusion simulations, including support for diffusion in intracellular and extracellular space


``Dendra`` is a research project and is still under development. If you have any questions, suggestions, or feedback, please let us know by opening an issue on our `GitLab repository <https://gitlab.oit.duke.edu/mah148/dendra>`_.

Getting started
---------------

``Dendra`` allows you to simulate the effects of stimulation on large numbers of biophysical neuron models on CPU or GPU:

.. code-block:: python

   import torch
   import matplotlib.pyplot as plt

   import dendra as dn
   from dendra_models.models import smolMRG
   from dendra.units import kHz, mA, ms, um

   # single 2.0 µm MRG model with extracellular stimulation
   model = smolMRG([2.0 * um], n_node=201)

   # Analytic point-source field in mV/mA, 200 µm from the axon.
   ve_s = dn.isotropic_point(z=200.0 * um)(model)

   dt, tstop = 0.001 * ms, 100 * ms
   f, amp = 5 * kHz, 0.5 * mA
   i_t = dn.sin(amp=amp, freq=f)

   # run simulation
   rec = dn.callbacks.Recorder(['v'], node_indices=model.c(0.9))
   model.steady_state()
   model.longrun(
      tstop=tstop, dt=dt, extra=(ve_s, i_t),
      chunklength=1000, callbacks=[rec], progressbar=True
   )

   # visualize
   v = rec.numpy('v')
   plt.plot(v[:, 0, 0]-v[0, 0, 0])
   plt.show()


Installation
------------

Dendra targets Python 3.11+ and PyTorch 2.8+ (CUDA 12.9+ wheels recommended for GPU use). A typical setup is:

1. Create and activate an isolated environment (optional): ``conda create -n dendra python=3.12 && conda activate dendra``.
2. Install PyTorch (choose GPU or CPU wheels): ``python -m pip install torch --index-url https://download.pytorch.org/whl/cu129``.
3. Clone the repo and install: ``git clone https://gitlab.oit.duke.edu/mah148/dendra.git && cd dendra && python -m pip install .``.

To build these docs locally, install the extras (``python -m pip install '.[doc]'``) and run ``make html`` inside ``docs``.

See :ref:`installation` for detailed guidance and optional extras (Jupyter, development tooling, and CPU implicit solver support via ``dendra-solvers``).


Feedback and Contributions
--------------------------

We welcome issues and pull requests on GitLab. When reporting a bug, include your OS, Python/PyTorch versions, install method, and a minimal reproducible script. For feature requests, please describe the workflow you are trying to support.

Contribution tips:

- Use a fresh branch and keep changes focused.
- Install development extras (``python -m pip install --editable '.[dev]'``) and run tests where applicable.
- Follow the existing style conventions; ``pre-commit`` hooks are configured for you (``python -m pre-commit install``).
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
