.. _advanced:

Guides
======

Find a workflow for building, running, or training a model. Use the
:doc:`API reference <dendra>` to look up functions, classes, and their arguments.

The `examples directory <https://github.com/wmglab-duke/dendra/tree/main/examples>`_
groups runnable scripts by task, with a guide in each folder. Start with
``examples/getting_started`` for small Population and Network workflows.

.. toctree::
   :hidden:
   :maxdepth: 2

   advanced/run-and-record-simulations
   advanced/build-models
   advanced/train-models
   advanced/performance-and-diagnostics

.. _simulation-training-and-observations:

Run and record simulations
--------------------------

:doc:`Run and record a simulation <advanced/run-and-record-simulations>` with
explicit inputs, repeatable randomness, and checkpoints you can resume.

.. _building-models:

Build models
------------

:doc:`Build a model <advanced/build-models>` by configuring parameters,
connecting populations, or defining your own mechanisms and synapses.

.. _specialized-training-and-diagnostics:

Train models
------------

:doc:`Train a model <advanced/train-models>` by selecting trainable parameters
and fitting them to a target trace or electrophysiological measurement.

Performance and diagnostics
---------------------------

:doc:`Batch and troubleshoot simulations <advanced/performance-and-diagnostics>`
using device placement and targeted model checks.
