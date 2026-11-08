Welcome to Dendra!
==================

Dendra simulates biophysical neurons and networks in PyTorch. Apply
intracellular currents or extracellular fields, record responses, and fit
parameters to voltage traces or physiological measurements.

Use built-in Hodgkin–Huxley and passive mechanisms or define your own
mechanisms and material processes. Connect cells with delayed events or
continuous signals. When training event-driven networks, you can fit connection
delays and use surrogate gradients for spike thresholds.

Installation
------------

Use Python 3.11 or newer in an isolated environment. On Windows, we recommend
using WSL2.

.. code-block:: console

   python -m pip install dendra

See the :ref:`installation guide <installation>` for PyTorch build selection,
optional native CPU and Apple Metal solvers, and installation checks.

Getting started
---------------

Stimulate one Hodgkin–Huxley compartment with a current pulse, then plot its
membrane voltage. The call to ``model.run`` advances the model in place.

.. code-block:: python

   import matplotlib.pyplot as plt

   import dendra as dn
   from dendra.models.mod import hh
   from dendra.units import ms, nA

   model = dn.SingleCompartment(N=1, C=1, celsius=6.3, cm=1.0, v_init=-65.0)
   model.diam.fill_(20.0)
   model.dx.fill_(20.0)
   model.insert(hh)
   model[..., 0].inject(
       dn.mono_rect(amp=0.1 * nA, delay=1.0 * ms, pw=1.0 * ms)
   )

   dt = 0.01 * ms
   duration = 10.0 * ms
   recorder = dn.Recorder(["t", "v"], node_indices=[0])
   model.initialize()
   model.run(duration=duration, dt=dt, callbacks=[recorder], progressbar=False)

   plt.plot(recorder.numpy("t"), recorder.numpy("v")[:, 0, 0])
   plt.xlabel("Time (ms)")
   plt.ylabel("Membrane potential (mV)")
   plt.show()

To run the same model through a reusable function, reset it to its initial
state and compile it. The ``eager`` backend keeps this first call short.

.. code-block:: python

   model.initialize()
   program = dn.compile(model, dt=dt, duration=duration, backend="eager")
   state, _ = program()

Continue with :ref:`basics` for stimulation, recording, and model construction.
The :ref:`functional execution guide <functional-execution>` shows how to
record outputs and continue a compiled run.

Feedback and Contributions
--------------------------

Report issues and open pull requests on
`GitHub <https://github.com/wmglab-duke/dendra>`_. For a bug report, include
your OS, Python and PyTorch versions, installation method, and a minimal
script. See the
`contribution guide <https://github.com/wmglab-duke/dendra/blob/main/CONTRIBUTING.md>`_
for development setup and testing.

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
   gpu-deployment
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
   testing
   dendra
   license
