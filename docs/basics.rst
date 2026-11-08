.. _basics:

Basics
======

- **Run your first model.** The :doc:`overview <basics/00_overview>` builds an
  axon, injects current, and records voltage.
- **Stimulate and record outside a cell.** Start with
  :doc:`extracellular stimulation and recording <basics/01a_extracellular_stim_and_rec>`,
  then try precomputed fields, multiple contacts, or an activation threshold.
- **Build and fit models.** Explore
  :doc:`branched morphologies <basics/02_branched_morphologies>`,
  :doc:`networks <basics/03_networks>`, or
  :doc:`parameter fitting <basics/04_gradient_descent>`.

Run the notebooks locally
-------------------------

After :ref:`installing Dendra <installation>`, install JupyterLab in the same
environment. Launch it from ``docs/basics`` in a repository checkout so the
notebooks can load the helper modules and data files next to them.

.. code-block:: console

   python -m pip install jupyterlab
   cd docs/basics
   python -m jupyter lab

- On CPU, the branched morphology and network simulations and the MRG axon
  examples in the thresholding notebook need the
  :ref:`native solver extra <installation>`.
- The MRG examples also use ``dendra-models``. Install the version used in these
  tutorials with ``python -m pip install "dendra-models==0.3.0"``.
- The multicontact tutorial's animation needs ``ffmpeg`` on your ``PATH``.

Tutorials
---------

.. toctree::
   :maxdepth: 1

   basics/00_overview.ipynb
   basics/01a_extracellular_stim_and_rec.ipynb
   basics/01b_precomputed_potentials.ipynb
   basics/01c_multicontact.ipynb
   basics/01d_thresholding.ipynb
   basics/02_branched_morphologies.ipynb
   basics/02a_native_morphologies.rst
   basics/03_networks.ipynb
   basics/04_gradient_descent.ipynb
