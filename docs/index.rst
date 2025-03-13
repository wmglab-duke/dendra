Welcome to AxonML!
===================

``AxonML`` is a differentiable simulator for biophysical neuron fiber models in `PyTorch <https://github.com/pytorch/pytorch>`_, with an emphasis on effects of extracellular stimulation. Its key features are:

- automatic differentiation, allowing gradient-based optimization of thousands of parameters  
- support for CPU and GPU with minimal changes to the code
- a wide range of popular biophysical models, including Hodgkin-Huxley, Tigerholm, MRG, and more
- ``jit``-compilation, making it blazing fast while being fully written in python  
- a simple API, making it easy to use for beginners and experts alike

``AxonML`` is a research project and is still under development. If you have any questions, suggestions, or feedback, please let us know by opening an issue on our `GitHub repository <https://github.com/wmglab-duke/axonml>`_.

Getting started
---------------

``AxonML`` allows you to simulate the effects of stimulation on large numbers of biophysical neuron models on CPU or GPU:

.. code-block:: python

    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    from axonml.models import *
    from axonml.models.callbacks import Recorder, LFP, Active

    # single 2.0 µm MRG model with extracellular stimulation
    model = smolMRG([2.0], n_comp=201)

    # point source extracellular kHz stimulation
    x = model.x()
    z = 100.0
    r = torch.sqrt(z**2 + x**2) * 1e-4
    ve_s = 1000 / (4 * torch.pi * 500 * r)

    dt = 0.001
    tstop = 100
    f = 5
    t = torch.arange(0, tstop, dt)
    amplitude = 25
    ve_t = amplitude * torch.sin(2 * torch.pi * f * t).unsqueeze(0)

    # run simulation
    rec = Recorder(['v'], node_indices=model.c(0.9))
    model.steady_state()
    model.longrun(space=ve_s, time=ve_t, n_chunks=10000, dt=dt, callbacks=[rec])

    # visualize
    v = rec.numpy('v')
    plt.plot(v[:, 0, 0, 0]-v[0, 0, 0, 0])
    plt.show()


Installation
------------

TODO: Add installation instructions.


Feedback and Contributions
--------------------------

TODO: Add information on how to contribute.


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