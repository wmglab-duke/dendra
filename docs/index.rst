Welcome to AxonML!
===================

``AxonML`` is a differentiable simulator for biophysical neuron fiber models in `PyTorch <https://github.com/pytorch/pytorch>`_, with an emphasis on effects of extracellular stimulation. Its key features are:

- support for CPU and GPU
- automatic differentiation, allowing gradient-based optimization of thousands of parameters
- implementations of a range of popular biophysical models, including Hodgkin-Huxley, Tigerholm, MRG, and more
- ``jit``-compilation, making it blazing fast while being fully written in python
- a simple API, making it easy to use for beginners and experts alike

``AxonML`` is a research project and is still under development. If you have any questions, suggestions, or feedback, please let us know by opening an issue on our `GitLab repository <https://gitlab.oit.duke.edu/mah148/axonml>`_.

Getting started
---------------

``AxonML`` allows you to simulate the effects of stimulation on large numbers of biophysical neuron models on CPU or GPU:

.. code-block:: python

    import torch
    import matplotlib.pyplot as plt

    import axonml as ax

    # single 2.0 µm MRG model with extracellular stimulation
    model = ax.smolMRG([2.0], n_node=201)

    # point source extracellular kHz stimulation
    ve_s = ax.isotropic_point(z=100.0, rhoe=500.0)(model)

    dt, tstop = 0.001, 100
    f, amp = 5, 0.5
    t = torch.arange(0, tstop, dt)
    i_t = ax.sin(amp=amp, freq=f)(t)

    # run simulation
    rec = ax.callbacks.Recorder(['v'], node_indices=model.c(0.9))
    model.steady_state()
    model.longrun(space=ve_s, time=i_t, n_chunks=10000, dt=dt, callbacks=[rec])

    # visualize
    v = rec.numpy('v')
    plt.plot(v[:, 0, 0, 0]-v[0, 0, 0, 0])
    plt.show()


Installation
------------

AxonML targets Python 3.11+ and PyTorch 2.7+ (CUDA 12.9 wheels recommended for GPU use). A typical setup is:

1. Create and activate an isolated environment (optional): ``conda create -n axonml python=3.12 && conda activate axonml``.
2. Install PyTorch (choose GPU or CPU wheels): ``python -m pip install torch --index-url https://download.pytorch.org/whl/cu129``.
3. Clone the repo and install: ``git clone https://gitlab.oit.duke.edu/mah148/axonml.git && cd axonml && python -m pip install .``.

To build these docs locally, install the extras (``python -m pip install '.[doc]'``) and run ``make html`` inside ``docs``.

See :ref:`installation` for detailed guidance and optional extras (Jupyter, development tooling, and CPU implicit solver support via ``axonml-solvers``).


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

If you use `AxonML`, consider citing the `corresponding paper <https://www.nature.com/articles/s41467-024-51709-8>`_:

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

.. toctree::
   :hidden:
   :maxdepth: 1

   basics

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Resources

   axonml
